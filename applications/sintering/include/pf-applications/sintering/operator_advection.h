// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2026 by the hpsint authors
//
// This file is part of the hpsint library.
//
// The hpsint library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hpsint.
//
// ---------------------------------------------------------------------

#pragma once

#include <pf-applications/lac/dynamic_block_vector.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/instantiation.h>
#include <pf-applications/sintering/operator_sintering_data.h>

#include <pf-applications/grain_tracker/tracker.h>
#include <pf-applications/matrix_free/tools.h>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <unordered_map>

namespace Sintering
{
  using namespace dealii;

  template <typename Number>
  class AdvectionOperator
  {
  public:
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    virtual ~AdvectionOperator() = default;

    virtual void
    evaluate_forces(
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const = 0;
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class AdvectionOperatorGeneric : public AdvectionOperator<Number>
  {
  public:
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type = Number;

    using T = AdvectionOperatorGeneric<dim, Number, VectorizedArrayType>;

    // Force, torque and grain volume
    static constexpr unsigned int n_comp_total = (dim == 3 ? 6 : 3);

    AdvectionOperatorGeneric(
      const double                                           k,
      const double                                           cgb,
      const double                                           ceq,
      const double                                           smoothening,
      const MatrixFree<dim, Number, VectorizedArrayType>    &matrix_free,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const GrainTracker::Tracker<dim, Number>              &grain_tracker,
      AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism)
      : matrix_free(matrix_free)
      , timer(true)
      , do_timing(true)
      , k(k)
      , cgb(cgb)
      , ceq(ceq)
      , smoothening(smoothening)
      , data(data)
      , grain_tracker(grain_tracker)
      , advection_mechanism(advection_mechanism)
    {}

    // Same as above, but k is recomputed once per evaluate_forces() call by
    // invoking k_eval(current_time) (e.g. to apply the Arrhenius relation).
    AdvectionOperatorGeneric(
      std::function<double(const double)>                    k_eval,
      const double                                           cgb,
      const double                                           ceq,
      const double                                           smoothening,
      const MatrixFree<dim, Number, VectorizedArrayType>    &matrix_free,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const GrainTracker::Tracker<dim, Number>              &grain_tracker,
      AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism)
      : matrix_free(matrix_free)
      , timer(true)
      , do_timing(true)
      , k(0.)
      , cgb(cgb)
      , ceq(ceq)
      , smoothening(smoothening)
      , k_eval(std::move(k_eval))
      , data(data)
      , grain_tracker(grain_tracker)
      , advection_mechanism(advection_mechanism)
    {}

    ~AdvectionOperatorGeneric() override
    {}

    void
    evaluate_forces(
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const override
    {
      MyScope scope(timer, "advection_op_gen::evaluate_forces", do_timing);

      if (k_eval)
        k = k_eval(data.get_time());

      advection_mechanism.nullify_data(grain_tracker.n_segments());

      // We do not have an output vector
      BlockVectorType dummy(1);

#define OPERATION(c, d)                                  \
  MyMatrixFreeTools::cell_loop_wrapper(                  \
    matrix_free,                                         \
    &AdvectionOperatorGeneric::do_evaluate_forces<c, d>, \
    this,                                                \
    dummy,                                               \
    src,                                                 \
    pre_operation,                                       \
    post_operation);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

      // Perform global communication
      MPI_Allreduce(MPI_IN_PLACE,
                    advection_mechanism.get_grains_data().data(),
                    advection_mechanism.get_grains_data().size(),
                    Utilities::MPI::mpi_type_id_for_type<Number>,
                    MPI_SUM,
                    MPI_COMM_WORLD);
    }

    unsigned int
    n_grains() const
    {
      return data.n_components() - 2;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      (void)n_grains;
      return n_comp_total;
    }

  private:
    template <int n_comp, int n_grains>
    void
    do_evaluate_forces(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType                                    &dummy,
      const BlockVectorType                              &solution,
      const std::pair<unsigned int, unsigned int>        &range) const
    {
      (void)dummy;

      FECellIntegrator<dim, 2 + n_grains, Number, VectorizedArrayType> phi_sint(
        matrix_free);

      FECellIntegrator<dim,
                       AdvectionMechanism<dim, Number, VectorizedArrayType>::
                         n_comp_volume_force_torque,
                       Number,
                       VectorizedArrayType>
        phi_ft(matrix_free);

      VectorizedArrayType cgb_lim(cgb);
      VectorizedArrayType zeros(0.0);
      VectorizedArrayType ones(1.0);

      std::vector<unsigned int> &index_ptr =
        advection_mechanism.get_index_ptr();
      std::vector<unsigned int> &index_values =
        advection_mechanism.get_index_values();

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi_sint.reinit(cell);
          phi_sint.gather_evaluate(
            solution,
            EvaluationFlags::EvaluationFlags::values |
              EvaluationFlags::EvaluationFlags::gradients);

          phi_ft.reinit(cell);

          const auto grain_to_relevant_grain =
            data.get_grain_to_relevant_grain(cell);

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              if (grain_to_relevant_grain[ig] ==
                  static_cast<unsigned char>(255))
                continue;

              Point<dim, VectorizedArrayType> rc;

              std::vector<std::pair<unsigned int, unsigned int>> segments(
                matrix_free.n_active_entries_per_cell_batch(cell));

              unsigned int i = 0;

              for (; i < segments.size(); ++i)
                {
                  bool set_invalid = false;

                  const auto icell = matrix_free.get_cell_iterator(cell, i);
                  const auto cell_index = icell->global_active_cell_index();

                  const unsigned int particle_id =
                    grain_tracker.get_particle_index(ig, cell_index);

                  if (particle_id != numbers::invalid_unsigned_int)
                    {
                      const auto grain_and_segment =
                        grain_tracker.get_grain_and_segment(ig, particle_id);

                      if (grain_and_segment.first !=
                          numbers::invalid_unsigned_int)
                        {
                          const auto &rc_i = grain_tracker.get_segment_center(
                            grain_and_segment.first, grain_and_segment.second);

                          for (unsigned int d = 0; d < dim; ++d)
                            rc[d][i] = rc_i[d];

                          segments[i] = grain_and_segment;

                          index_values.push_back(
                            grain_tracker.get_grain_segment_index(
                              grain_and_segment.first,
                              grain_and_segment.second));
                        }
                      else
                        {
                          set_invalid = true;
                        }
                    }
                  else
                    {
                      set_invalid = true;
                    }

                  if (set_invalid)
                    {
                      segments[i] =
                        std::make_pair(numbers::invalid_unsigned_int,
                                       numbers::invalid_unsigned_int);

                      index_values.push_back(numbers::invalid_unsigned_int);
                    }
                }

              for (; i < VectorizedArrayType::size(); ++i)
                index_values.push_back(numbers::invalid_unsigned_int);

              for (unsigned int q = 0; q < phi_sint.n_q_points; ++q)
                {
                  const auto val  = phi_sint.get_value(q);
                  const auto grad = phi_sint.get_gradient(q);

                  auto &c          = val[0];
                  auto &eta_i      = val[2 + ig];
                  auto &eta_grad_i = grad[2 + ig];

                  const auto &r = phi_sint.quadrature_point(q);

                  Tensor<1,
                         AdvectionMechanism<dim, Number, VectorizedArrayType>::
                           n_comp_volume_force_torque,
                         VectorizedArrayType>
                                                      value_result;
                  Tensor<1, dim, VectorizedArrayType> force;
                  moment_t<dim, VectorizedArrayType>  torque;
                  torque = 0;

                  // Compute force and torque acting on grain i from each of the
                  // other grains
                  for (unsigned int jg = 0; jg < n_grains; ++jg)
                    {
                      if (grain_to_relevant_grain[jg] ==
                          static_cast<unsigned char>(255))
                        continue;

                      if (ig != jg)
                        {
                          auto &eta_j      = val[2 + jg];
                          auto &eta_grad_j = grad[2 + jg];

                          // Vector normal to the grain boundary
                          Tensor<1, dim, VectorizedArrayType> dF =
                            eta_grad_i - eta_grad_j;

                          // Filter to detect grain boundary
                          auto etai_etaj = eta_i * eta_j;

                          // Normalize or not
                          if (cgb > 0)
                            {
                              if (smoothening > 0)
                                etai_etaj =
                                  ones /
                                  (ones + std::exp(-smoothening *
                                                   (etai_etaj - cgb_lim)));
                              else
                                etai_etaj = compare_and_apply_mask<
                                  SIMDComparison::greater_than>(etai_etaj,
                                                                cgb_lim,
                                                                ones,
                                                                zeros);
                            }
                          else
                            {
                              etai_etaj *= 4.;
                            }

                          // Compute force component per cell
                          dF *= k * (c - ceq) * etai_etaj;

                          force += dF;

                          // Vector pointing from the grain center to the
                          // current qp point
                          const auto r_rc = (r - rc);

                          // Torque as cross product
                          // (scalar in 2D and vector in 3D)
                          torque += cross_product(r_rc, dF);
                        }
                    }

                  // Volume of grain i
                  value_result[0] = eta_i;

                  // Force acting on grain i
                  for (unsigned int d = 0; d < dim; ++d)
                    value_result[d + 1] = force[d];

                  // Torque acting on grain i
                  if constexpr (moment_s<dim, VectorizedArrayType> == 1)
                    value_result[dim + 1] = torque;
                  else
                    for (unsigned int d = 0;
                         d < moment_s<dim, VectorizedArrayType>;
                         ++d)
                      value_result[d + dim + 1] = torque[d];

                  phi_ft.submit_value(value_result, q);
                }

              const auto volume_force_torque = phi_ft.integrate_value();

              for (unsigned int i = 0; i < segments.size(); ++i)
                {
                  const auto &grain_and_segment = segments[i];

                  if (grain_and_segment.first != numbers::invalid_unsigned_int)
                    {
                      const auto segment_index =
                        grain_tracker.get_grain_segment_index(
                          grain_and_segment.first, grain_and_segment.second);

                      const auto &rc_i = grain_tracker.get_segment_center(
                        grain_and_segment.first, grain_and_segment.second);

                      for (unsigned int d = 0; d < dim; ++d)
                        advection_mechanism.grain_center(segment_index)[d] +=
                          rc_i[d];

                      for (unsigned int d = 0;
                           d < advection_mechanism.n_comp_volume_force_torque;
                           ++d)
                        advection_mechanism.grain_data(segment_index)[d] +=
                          volume_force_torque[d][i];
                    }
                }
            }

          AssertDimension(index_values.size() % VectorizedArrayType::size(), 0);

          index_ptr.push_back(index_values.size());
        }
    }

    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

    mutable MyTimerOutput timer;
    mutable bool          do_timing;

    mutable double k;
    const double   cgb;
    const double   ceq;
    const double   smoothening;

    // Optional hook to recompute k once per evaluate_forces() call (e.g. via
    // the Arrhenius relation). Left empty by default, in which case k stays
    // constant.
    std::function<double(const double)> k_eval;

    const SinteringOperatorData<dim, VectorizedArrayType> &data;
    const GrainTracker::Tracker<dim, Number>              &grain_tracker;

    AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism;
  };

  template <int dim, typename Number, typename VectorizedArrayType>
  class AdvectionOperatorWeighted : public AdvectionOperator<Number>
  {
  public:
    using T = AdvectionOperatorWeighted<dim, Number, VectorizedArrayType>;

    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type = Number;

    // Some required number of components
    static constexpr unsigned int n_comp_volume   = 1;
    static constexpr unsigned int n_comp_force_gb = dim + 1;
    static constexpr unsigned int n_comp_total    = n_comp_force_gb;

    AdvectionOperatorWeighted(
      const double                                           k,
      const double                                           cgb,
      const double                                           ceq,
      const double                                           smoothening,
      const MatrixFree<dim, Number, VectorizedArrayType>    &matrix_free,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const GrainTracker::Tracker<dim, Number>              &grain_tracker,
      AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism)
      : matrix_free(matrix_free)
      , timer(true)
      , do_timing(true)
      , k(k)
      , cgb(cgb)
      , ceq(ceq)
      , smoothening(smoothening)
      , data(data)
      , grain_tracker(grain_tracker)
      , advection_mechanism(advection_mechanism)
    {}

    // Same as above, but k is recomputed once per evaluate_forces() call by
    // invoking k_eval(current_time) (e.g. to apply the Arrhenius relation).
    AdvectionOperatorWeighted(
      std::function<double(const double)>                    k_eval,
      const double                                           cgb,
      const double                                           ceq,
      const double                                           smoothening,
      const MatrixFree<dim, Number, VectorizedArrayType>    &matrix_free,
      const SinteringOperatorData<dim, VectorizedArrayType> &data,
      const GrainTracker::Tracker<dim, Number>              &grain_tracker,
      AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism)
      : matrix_free(matrix_free)
      , timer(true)
      , do_timing(true)
      , k(0.)
      , cgb(cgb)
      , ceq(ceq)
      , smoothening(smoothening)
      , k_eval(std::move(k_eval))
      , data(data)
      , grain_tracker(grain_tracker)
      , advection_mechanism(advection_mechanism)
    {}

    ~AdvectionOperatorWeighted() override
    {}

    void
    evaluate_forces(
      const BlockVectorType &src,
      const std::function<void(const unsigned int, const unsigned int)>
        pre_operation = {},
      const std::function<void(const unsigned int, const unsigned int)>
        post_operation = {}) const override
    {
      MyScope scope(timer, "advection_op_wgt::evaluate_forces", do_timing);

      if (k_eval)
        k = k_eval(data.get_time());

      const auto n_segments = grain_tracker.n_segments();

      {
        MyScope scope(timer,
                      "advection_op_wgt::evaluate_forces::init",
                      do_timing);

        advection_mechanism.nullify_data(n_segments);

        pair_data.clear();
        grains_data_volumes.assign(n_segments, 0.);
      }

      // We do not have an output vector
      BlockVectorType dummy(1);

      {
        MyScope scope(timer,
                      "advection_op_wgt::evaluate_forces::cell_loop",
                      do_timing);

#define OPERATION(c, d)                                   \
  MyMatrixFreeTools::cell_loop_wrapper(                   \
    matrix_free,                                          \
    &AdvectionOperatorWeighted::do_evaluate_forces<c, d>, \
    this,                                                 \
    dummy,                                                \
    src,                                                  \
    pre_operation,                                        \
    post_operation);
        EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
      }

      MyScope scope_reduce(timer,
                           "advection_op_wgt::evaluate_forces::allreduce",
                           do_timing);

      // Gather the locally touched segment pairs from all ranks and sum them,
      // in rank order so that all ranks get identical results
      std::map<std::uint64_t, std::array<Number, n_comp_force_gb>>
        global_pair_data;
      {
        std::vector<std::uint64_t> keys;
        std::vector<Number>        values;
        keys.reserve(pair_data.size());
        values.reserve(n_comp_force_gb * pair_data.size());
        for (const auto &[key, value] : pair_data)
          {
            keys.push_back(key);
            values.insert(values.end(), value.begin(), value.end());
          }

        const int  n_local = keys.size();
        const auto n_ranks = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

        std::vector<int> counts(n_ranks);
        MPI_Allgather(
          &n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> displs(n_ranks, 0);
        for (unsigned int r = 1; r < n_ranks; ++r)
          displs[r] = displs[r - 1] + counts[r - 1];
        const int n_global = displs.back() + counts.back();

        std::vector<std::uint64_t> all_keys(n_global);
        MPI_Allgatherv(keys.data(),
                       n_local,
                       MPI_UINT64_T,
                       all_keys.data(),
                       counts.data(),
                       displs.data(),
                       MPI_UINT64_T,
                       MPI_COMM_WORLD);

        for (unsigned int r = 0; r < n_ranks; ++r)
          {
            counts[r] *= n_comp_force_gb;
            displs[r] *= n_comp_force_gb;
          }

        std::vector<Number> all_values(n_comp_force_gb * n_global);
        MPI_Allgatherv(values.data(),
                       n_comp_force_gb * n_local,
                       Utilities::MPI::mpi_type_id_for_type<Number>,
                       all_values.data(),
                       counts.data(),
                       displs.data(),
                       Utilities::MPI::mpi_type_id_for_type<Number>,
                       MPI_COMM_WORLD);

        for (int e = 0; e < n_global; ++e)
          {
            auto &entry =
              global_pair_data
                .try_emplace(all_keys[e], std::array<Number, n_comp_force_gb>{})
                .first->second;
            for (unsigned int c = 0; c < n_comp_force_gb; ++c)
              entry[c] += all_values[n_comp_force_gb * e + c];
          }
      }

      // Perform global communication
      MPI_Allreduce(MPI_IN_PLACE,
                    grains_data_volumes.data(),
                    grains_data_volumes.size(),
                    Utilities::MPI::mpi_type_id_for_type<Number>,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      // Pair entries (a, b) with a < b hold F_ab, and F_ba = -F_ab
      std::vector<Tensor<1, dim, Number>> forces(n_segments);
      for (const auto &[key, value] : global_pair_data)
        {
          const unsigned int a     = key >> 32;
          const unsigned int b     = key & 0xffffffff;
          const auto         gb_ab = value[dim];

          if (std::abs(gb_ab) > 1e-10)
            for (unsigned int d = 0; d < dim; ++d)
              {
                forces[a][d] += value[d] / gb_ab;
                forces[b][d] -= value[d] / gb_ab;
              }
        }

      // Communicate the forces to the regular advection, the index data
      // filled in the cell loop has to be kept
      for (unsigned int i = 0; i < n_segments; ++i)
        {
          advection_mechanism.grain_data(i)[0] = grains_data_volumes[i];

          for (unsigned int d = 0; d < dim; ++d)
            advection_mechanism.grain_data(i)[d + 1] = forces[i][d];

          for (unsigned int d = 1 + dim;
               d < AdvectionMechanism<dim, Number, VectorizedArrayType>::
                     n_comp_volume_force_torque;
               ++d)
            advection_mechanism.grain_data(i)[d] = 0.;
        }
    }

    unsigned int
    n_grains() const
    {
      return data.n_components() - 2;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      (void)n_grains;
      return n_comp_total;
    }

  private:
    template <int n_comp, int n_grains>
    void
    do_evaluate_forces(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType                                    &dummy,
      const BlockVectorType                              &solution,
      const std::pair<unsigned int, unsigned int>        &range) const
    {
      (void)dummy;

      FECellIntegrator<dim, 2 + n_grains, Number, VectorizedArrayType> phi_sint(
        matrix_free);

      FECellIntegrator<dim, n_comp_volume, Number, VectorizedArrayType> phi_v(
        matrix_free);

      FECellIntegrator<dim, n_comp_force_gb, Number, VectorizedArrayType>
        phi_fg(matrix_free);

      VectorizedArrayType cgb_lim(cgb);
      VectorizedArrayType zeros(0.0);
      VectorizedArrayType ones(1.0);

      std::vector<unsigned int> &index_ptr =
        advection_mechanism.get_index_ptr();
      std::vector<unsigned int> &index_values =
        advection_mechanism.get_index_values();

      constexpr unsigned int n_lanes = VectorizedArrayType::size();

      // Pairs with max(eta_i) * max(eta_j) below this value within a cell
      // are skipped, the check is done per cell (lane) to make the result
      // independent of the cell batching
      constexpr Number pair_tolerance = 1e-12;

      // Global segment index of each lane, done once per grain and cell
      const auto fill_segment_indices =
        [&, this](const auto                         cell,
                  const unsigned int                 ig,
                  std::array<unsigned int, n_lanes> &segment_indices) {
          segment_indices.fill(numbers::invalid_unsigned_int);

          for (unsigned int i = 0;
               i < matrix_free.n_active_entries_per_cell_batch(cell);
               ++i)
            {
              const auto icell      = matrix_free.get_cell_iterator(cell, i);
              const auto cell_index = icell->global_active_cell_index();

              const unsigned int particle_id =
                grain_tracker.get_particle_index(ig, cell_index);

              if (particle_id == numbers::invalid_unsigned_int)
                continue;

              const auto grain_and_segment =
                grain_tracker.get_grain_and_segment(ig, particle_id);

              if (grain_and_segment.first != numbers::invalid_unsigned_int)
                segment_indices[i] = grain_tracker.get_grain_segment_index(
                  grain_and_segment.first, grain_and_segment.second);
            }

          index_values.insert(index_values.end(),
                              segment_indices.begin(),
                              segment_indices.end());
        };

      std::array<std::array<unsigned int, n_lanes>, n_grains> segment_indices;
      std::array<VectorizedArrayType, n_grains>               eta_max;

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          phi_sint.reinit(cell);
          phi_sint.gather_evaluate(
            solution,
            EvaluationFlags::EvaluationFlags::values |
              EvaluationFlags::EvaluationFlags::gradients);

          phi_v.reinit(cell);
          phi_fg.reinit(cell);

          const auto grain_to_relevant_grain =
            data.get_grain_to_relevant_grain(cell);

          const auto is_relevant = [&](const unsigned int ig) {
            return grain_to_relevant_grain[ig] !=
                   static_cast<unsigned char>(255);
          };

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              if (!is_relevant(ig))
                continue;

              fill_segment_indices(cell, ig, segment_indices[ig]);

              eta_max[ig] = 0.;
              for (unsigned int q = 0; q < phi_sint.n_q_points; ++q)
                eta_max[ig] = std::max(eta_max[ig],
                                       std::abs(phi_sint.get_value(q)[2 + ig]));

              // Deal with volume only first
              for (unsigned int q = 0; q < phi_sint.n_q_points; ++q)
                {
                  Tensor<1, 1, VectorizedArrayType> value_result;
                  value_result[0] = phi_sint.get_value(q)[2 + ig];

                  phi_v.submit_value(value_result, q);
                }

              const auto volume = phi_v.integrate_value();

              for (unsigned int v = 0; v < n_lanes; ++v)
                if (segment_indices[ig][v] != numbers::invalid_unsigned_int)
                  grains_data_volumes[segment_indices[ig][v]] += volume[v];
            }

          // Now deal with forces, F_ji = -F_ij and the GB area is symmetric
          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              if (!is_relevant(ig))
                continue;

              for (unsigned int jg = ig + 1; jg < n_grains; ++jg)
                {
                  if (!is_relevant(jg))
                    continue;

                  const auto eta_max_ij = eta_max[ig] * eta_max[jg];

                  bool pair_active = false;
                  for (unsigned int v = 0; v < n_lanes; ++v)
                    pair_active |= eta_max_ij[v] >= pair_tolerance;

                  if (!pair_active)
                    continue;

                  for (unsigned int q = 0; q < phi_sint.n_q_points; ++q)
                    {
                      const auto val  = phi_sint.get_value(q);
                      const auto grad = phi_sint.get_gradient(q);

                      auto &c          = val[0];
                      auto &eta_i      = val[2 + ig];
                      auto &eta_grad_i = grad[2 + ig];

                      auto &eta_j      = val[2 + jg];
                      auto &eta_grad_j = grad[2 + jg];

                      Tensor<1, n_comp_force_gb, VectorizedArrayType>
                        value_result;

                      // Vector normal to the grain boundary
                      Tensor<1, dim, VectorizedArrayType> dF =
                        eta_grad_i - eta_grad_j;

                      // Filter to detect grain boundary
                      auto etai_etaj = eta_i * eta_j;

                      // Normalize or not
                      if (cgb > 0)
                        {
                          if (smoothening > 0)
                            etai_etaj =
                              ones / (ones + std::exp(-smoothening *
                                                      (etai_etaj - cgb_lim)));
                          else
                            etai_etaj = compare_and_apply_mask<
                              SIMDComparison::greater_than>(etai_etaj,
                                                            cgb_lim,
                                                            ones,
                                                            zeros);
                        }
                      else
                        {
                          etai_etaj *= 4.;
                        }

                      // Compute force component per cell
                      dF *= k * (c - ceq) * etai_etaj;

                      VectorizedArrayType gb_area = eta_i * eta_j;

                      // Force acting between grains i and j
                      for (unsigned int d = 0; d < dim; ++d)
                        value_result[d] = dF[d];

                      // Store GB area
                      value_result[dim] = gb_area;

                      phi_fg.submit_value(value_result, q);
                    }

                  const auto force_gb = phi_fg.integrate_value();

                  for (unsigned int v = 0; v < n_lanes; ++v)
                    {
                      const auto segment_index_i = segment_indices[ig][v];
                      const auto segment_index_j = segment_indices[jg][v];

                      if (segment_index_i == numbers::invalid_unsigned_int ||
                          segment_index_j == numbers::invalid_unsigned_int ||
                          segment_index_i == segment_index_j ||
                          eta_max_ij[v] < pair_tolerance)
                        continue;

                      // Store F_ab for a < b only, F_ba = -F_ab
                      const bool   ordered = segment_index_i < segment_index_j;
                      const Number sign    = ordered ? 1. : -1.;
                      const auto   key =
                        (std::uint64_t(ordered ? segment_index_i :
                                                 segment_index_j)
                         << 32) |
                        (ordered ? segment_index_j : segment_index_i);

                      auto &entry =
                        pair_data
                          .try_emplace(key,
                                       std::array<Number, n_comp_force_gb>{})
                          .first->second;

                      for (unsigned int d = 0; d < dim; ++d)
                        entry[d] += sign * force_gb[d][v];

                      entry[dim] += force_gb[dim][v];
                    }
                }
            }

          AssertDimension(index_values.size() % VectorizedArrayType::size(), 0);

          index_ptr.push_back(index_values.size());
        }
    }

    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free;

    mutable MyTimerOutput timer;
    mutable bool          do_timing;

    mutable double k;
    const double   cgb;
    const double   ceq;
    const double   smoothening;

    // Optional hook to recompute k once per evaluate_forces() call (e.g. via
    // the Arrhenius relation). Left empty by default, in which case k stays
    // constant.
    std::function<double(const double)> k_eval;

    const SinteringOperatorData<dim, VectorizedArrayType> &data;
    const GrainTracker::Tracker<dim, Number>              &grain_tracker;

    AdvectionMechanism<dim, Number, VectorizedArrayType> &advection_mechanism;

    // Force and GB area of the touching segment pairs (a, b), a < b, keyed by
    // (a << 32 | b)
    mutable std::unordered_map<std::uint64_t,
                               std::array<Number, n_comp_force_gb>>
                                pair_data;
    mutable std::vector<Number> grains_data_volumes;
  };
} // namespace Sintering
