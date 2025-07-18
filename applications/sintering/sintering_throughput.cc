// ---------------------------------------------------------------------
//
// Copyright (C) 2022 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Test performance of the sintering operator

#ifndef SINTERING_DIM
static_assert(false, "No dimension has been given!");
#endif

#ifndef MAX_SINTERING_GRAINS
static_assert(false, "No grains number has been given!");
#endif

// #define USE_FE_Q_iso_Q1
// #define WITH_TENSORIAL_MOBILITY

#ifdef USE_FE_Q_iso_Q1
#  define FE_DEGREE 2
#  define N_Q_POINTS_1D FE_DEGREE * 2
#else
#  define FE_DEGREE 1
#  define N_Q_POINTS_1D FE_DEGREE + 1
#endif

#define WITH_TIMING
// #define WITH_TIMING_OUTPUT

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/revision.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

#include <pf-applications/base/performance.h>
#include <pf-applications/base/revision.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/free_energy.h>
#include <pf-applications/sintering/mobility.h>
#include <pf-applications/sintering/operator_sintering_data.h>
#include <pf-applications/sintering/operator_sintering_generic.h>
#include <pf-applications/sintering/preconditioners.h>

#include <pf-applications/time_integration/time_schemes.h>

using namespace dealii;
using namespace Sintering;
using namespace TimeIntegration;

template <typename T, typename VectorType>
using evaluate_nonlinear_residual_t =
  decltype(std::declval<const T>().evaluate_nonlinear_residual(
    std::declval<VectorType &>(),
    std::declval<const VectorType &>()));

template <typename T, typename VectorType>
constexpr bool has_evaluate_nonlinear_residual = dealii::internal::
  is_supported_operation<evaluate_nonlinear_residual_t, T, VectorType>;


// clang-format off
/**
 * likwid-mpirun -np 40 -f -g CACHES   -m -O ./applications/sintering/sintering-throughput
 * likwid-mpirun -np 40 -f -g FLOPS_DP -m -O ./applications/sintering/sintering-throughput
 */
// clang-format on
int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  const unsigned int dim                  = SINTERING_DIM;
  const unsigned int fe_degree            = 1;
  unsigned int       n_global_refinements = 7;
  const unsigned int max_sintering_grains = std::min(12, MAX_SINTERING_GRAINS);
  const unsigned int max_sintering_grains_mb =
    std::min(4u, max_sintering_grains);
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;
  using VectorType          = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType =
    LinearAlgebra::distributed::DynamicBlockVector<Number>;

  const bool scalar_mobility =
    SinteringOperatorData<dim, VectorizedArrayType>::use_tensorial_mobility ==
    false;

  constexpr bool test_helmholtz         = true;
  constexpr bool test_sintering_generic = true;
  constexpr bool test_sintering_wang    = true & scalar_mobility;

  // some arbitrary constants
  const double A       = 16;
  const double B       = 1;
  const double kappa_c = 1;
  const double kappa_p = 0.5;
  const double Mvol    = 1e-2;
  const double Mvap    = 1e-10;
  const double Msurf   = 4;
  const double Mgb     = 0.4;
  const double L       = 1;
  const double mt      = 1.0;
  const double mr      = 1.0;
  const double t       = 0.0;
  const double dt      = 0.1;

  const std::string fe_type =
#ifdef USE_FE_Q_iso_Q1
    "FE_Q_iso_Q1";
#else
    "FE_Q";
#endif

  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>>    quadrature;
  MappingQ1<dim>                      mapping;

  if (fe_type == "FE_Q")
    {
      fe = std::make_unique<FE_Q<dim>>(fe_degree);

      quadrature = std::make_unique<QGauss<dim>>(fe_degree + 1);
    }
  else if (fe_type == "FE_Q_iso_Q1")
    {
      AssertThrow(fe_degree == 1, ExcInternalError());

      const unsigned int n_subdivisions = 2;

      fe = std::make_unique<FE_Q_iso_Q1<dim>>(n_subdivisions);
      quadrature =
        std::make_unique<QIterated<dim>>(QGauss<1>(2), n_subdivisions);

      n_global_refinements -= 1;
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(*fe);

  AffineConstraints<Number> constraints;

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_quadrature_points;
  additional_data.overlap_communication_computation = false;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, *quadrature, additional_data);

  ConvergenceTable table;

  const auto test_operator = [&](const auto &op, const std::string label) {
    if constexpr (has_evaluate_nonlinear_residual<
                    typename std::remove_cv<
                      typename std::remove_reference<decltype(op)>::type>::type,
                    BlockVectorType>) // ... matrix-free -> rhs
      {
        BlockVectorType src, dst;
        op.initialize_dof_vector(src);
        op.initialize_dof_vector(dst);
        src = 1.0;

        const auto time = run_measurement(
          [&]() { op.template evaluate_nonlinear_residual<1>(dst, src); });

        table.add_value("t_" + label + "_rhs", time);
        table.set_scientific("t_" + label + "_rhs", true);
      }

    if (true) // ... matrix-free -> Jacobian
      {
        BlockVectorType src, dst;
        op.initialize_dof_vector(src);
        op.initialize_dof_vector(dst);
        src = 1.0;

        const auto time = run_measurement([&]() { op.vmult(dst, src); });

        table.add_value("t_" + label + "_mf", time);
        table.set_scientific("t_" + label + "_mf", true);
      }

    if (op.n_components() <= 2 + max_sintering_grains_mb) // ... matrix-based
      {
        op.initialize_system_matrix();

        const auto &matrix = op.get_system_matrix();

        const auto partitioner = op.get_system_partitioner();

        VectorType src, dst;
        src.reinit(partitioner);
        dst.reinit(partitioner);
        src = 1.0;

        const auto time = run_measurement([&]() { matrix.vmult(dst, src); });

        table.add_value("t_" + label + "_mb", time);
        table.set_scientific("t_" + label + "_mb", true);
        table.add_value("nnz_" + label, matrix.n_nonzero_elements());
      }
    else
      {
        table.add_value("t_" + label + "_mb", 0.0);
        table.set_scientific("t_" + label + "_mb", true);
        table.add_value("nnz_" + label, 0);
      }
  };

  const auto test_operator_dummy = [&](const std::string label) {
    table.add_value("t_" + label + "_rhs", 0);
    table.set_scientific("t_" + label + "_rhs", true);
    table.add_value("t_" + label + "_mf", 0);
    table.set_scientific("t_" + label + "_mf", true);
    table.add_value("t_" + label + "_mb", 0);
    table.set_scientific("t_" + label + "_mb", true);
    table.add_value("nnz_" + label, 0);
  };

  for (unsigned int n_components = 1; n_components <= max_sintering_grains + 2;
       ++n_components)
    {
      table.add_value("dim", dim);
      table.add_value("fe_type", fe_type);
      table.add_value("n_dofs", dof_handler.n_dofs());
      table.add_value("n_components", n_components);

      if constexpr (test_helmholtz) // test Helmholtz operator
        {
          HelmholtzOperator<dim, Number, VectorizedArrayType>
            helmholtz_operator(matrix_free, constraints, n_components);

          test_operator(helmholtz_operator, "helmholtz");
        }

      if constexpr (test_sintering_generic)
        {
          if (n_components >= 3) // test sintering operator
            {
              const std::shared_ptr<MobilityProvider> mobility_provider =
                std::make_shared<ProviderAbstract>(Mvol, Mvap, Msurf, Mgb, L);

              FreeEnergy<VectorizedArrayType> free_energy(A, B);

              TimeIntegratorData<Number> time_data(
                std::make_unique<BDF1Scheme<Number>>(), dt);

              SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
                kappa_c, kappa_p, mobility_provider, std::move(time_data));

              SolutionHistory<BlockVectorType> solution_history(
                sintering_data.time_data.get_order() + 1);

              sintering_data.set_n_components(n_components);
              sintering_data.set_time(t);

              AdvectionMechanism<dim, Number, VectorizedArrayType>
                advection_mechanism;

              const SinteringOperatorGeneric<dim, Number, VectorizedArrayType>
                sintering_operator(matrix_free,
                                   constraints,
                                   free_energy,
                                   sintering_data,
                                   solution_history,
                                   advection_mechanism,
                                   false,
                                   true);

              BlockVectorType src;
              sintering_operator.initialize_dof_vector(src);
              src = 1.0;

              sintering_data.fill_quadrature_point_values(matrix_free,
                                                          src,
                                                          false,
                                                          false);


              test_operator(sintering_operator, "sintering");
            }
          else
            {
              test_operator_dummy("sintering");
            }
        }

      if constexpr (test_sintering_wang)
        {
          if (n_components >= 3) // test wang operator
            {
              const std::shared_ptr<MobilityProvider> mobility_provider =
                std::make_shared<ProviderAbstract>(Mvol, Mvap, Msurf, Mgb, L);

              FreeEnergy<VectorizedArrayType> free_energy(A, B);

              TimeIntegratorData<Number> time_data(
                std::make_unique<BDF1Scheme<Number>>(), dt);

              SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
                kappa_c, kappa_p, mobility_provider, std::move(time_data));

              SolutionHistory<BlockVectorType> solution_history(
                sintering_data.time_data.get_order() + 1);


              sintering_data.set_n_components(n_components);
              sintering_data.set_time(t);

              const unsigned int n_grains = n_components - 2;

              // Dummy advection data
              AdvectionMechanism<dim, Number, VectorizedArrayType>
                advection_mechanism(true, mt, mr);

              // Dummy data - set a single segment for all cells
              const auto n_segments = 1;
              advection_mechanism.nullify_data(n_segments);

              auto gdata     = advection_mechanism.grain_data(0);
              gdata[0]       = 1.0; // volume
              gdata[1]       = 1.0; // force
              gdata[dim + 1] = 1.0; // torque

              // Set up indices
              const auto index_increment =
                n_grains * VectorizedArrayType::size();
              const auto n_index_values = n_grains *
                                          VectorizedArrayType::size() *
                                          matrix_free.n_cell_batches();

              for (unsigned int i = 0; i < matrix_free.n_cell_batches(); ++i)
                advection_mechanism.get_index_ptr().push_back(
                  advection_mechanism.get_index_ptr().back() + index_increment);
              advection_mechanism.get_index_values().assign(n_index_values, 0);

              const SinteringOperatorGeneric<dim, Number, VectorizedArrayType>
                sintering_operator(matrix_free,
                                   constraints,
                                   free_energy,
                                   sintering_data,
                                   solution_history,
                                   advection_mechanism,
                                   false,
                                   true);

              BlockVectorType src;
              sintering_operator.initialize_dof_vector(src);
              src = 1.0;

              sintering_data.fill_quadrature_point_values(
                matrix_free, src, true /*TODO: gradient*/, false);


              test_operator(sintering_operator, "wang");
            }
          else
            {
              test_operator_dummy("wang");
            }
        }
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
