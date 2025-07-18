// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the hpsint authors
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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include "operators.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// #define WITH_TENSORIAL_MOBILITY

#define MAX_SINTERING_GRAINS 10
#define MAX_N_COMPONENTS MAX_SINTERING_GRAINS + 2

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/free_energy.h>
#include <pf-applications/sintering/mobility.h>
#include <pf-applications/sintering/operator_sintering_data.h>
#include <pf-applications/sintering/operator_sintering_generic.h>

#ifdef LIKWID_PERFMON
static unsigned int likwid_counter = 0;
#endif

using namespace dealii;
using namespace Sintering;
using namespace TimeIntegration;

struct Parameters
{
  unsigned int dim                  = 2;
  unsigned int n_global_refinements = 1;

  unsigned int fe_degree           = 2;
  unsigned int n_quadrature_points = 0;
  unsigned int n_subdivisions      = 1;
  std::string  fe_type             = "FE_Q";

  unsigned int level = 2;

  unsigned int n_repetitions = 10;

  void
  parse(const std::string file_name)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);

    prm.parse_input(file_name, "", true);
  }

private:
  void
  add_parameters(ParameterHandler &prm)
  {
    prm.add_parameter("dim", dim);
    prm.add_parameter("n global refinements", n_global_refinements);

    prm.add_parameter("fe type",
                      fe_type,
                      "",
                      Patterns::Selection("FE_Q|FE_Q_iso_Q1"));
    prm.add_parameter("fe degree", fe_degree);
    prm.add_parameter("n quadrature points", n_quadrature_points);
    prm.add_parameter("n subdivisions", n_subdivisions);

    prm.add_parameter("level", level);

    prm.add_parameter("n repetitions", n_repetitions);
  }
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(const unsigned int component)
    : component(component)
  {}


  virtual double
  value(const Point<dim> &p, const unsigned int = 0) const override
  {
    if (component == 0)
      return p[0];
    else
      return 0.0;
  }

private:
  const unsigned int component;
};

template <int n_components,
          int dim,
          typename Number,
          typename VectorizedArrayType,
          typename QPointType>
std::shared_ptr<ProjectionOperatorBase<Number>>
create_op(const unsigned int                                  level,
          const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
          const QPointType                                   &q_point_operator)
{
  const auto &si = matrix_free.get_shape_info().data.front();

  const unsigned int fe_degree     = si.fe_degree;
  const unsigned int n_q_points_1d = si.n_q_points_1d;

  if ((fe_degree == 1) && (n_q_points_1d == 2))
    return std::make_shared<ProjectionOperator<dim,
                                               1,
                                               2,
                                               n_components,
                                               Number,
                                               VectorizedArrayType,
                                               QPointType>>(matrix_free,
                                                            q_point_operator,
                                                            level);
  if ((fe_degree == 2) && (n_q_points_1d == 4))
    return std::make_shared<ProjectionOperator<dim,
                                               2,
                                               4,
                                               n_components,
                                               Number,
                                               VectorizedArrayType,
                                               QPointType>>(matrix_free,
                                                            q_point_operator,
                                                            level);
  if ((fe_degree == 3) && (n_q_points_1d == 6))
    return std::make_shared<ProjectionOperator<dim,
                                               3,
                                               6,
                                               n_components,
                                               Number,
                                               VectorizedArrayType,
                                               QPointType>>(matrix_free,
                                                            q_point_operator,
                                                            level);

  AssertThrow(false, ExcNotImplemented());

  return std::make_shared<ProjectionOperator<dim,
                                             -1,
                                             0,
                                             n_components,
                                             Number,
                                             VectorizedArrayType,
                                             QPointType>>(matrix_free,
                                                          q_point_operator,
                                                          level);
}

template <int dim,
          typename Number,
          typename VectorizedArrayType,
          typename QPointType>
std::shared_ptr<ProjectionOperatorBase<Number>>
create_op(const unsigned int                                  n_components,
          const unsigned int                                  level,
          const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
          const QPointType                                   &q_point_operator)
{
#if MAX_N_COMPONENTS >= 1
  if (n_components == 1)
    return create_op<1>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 2
    if (n_components == 2)
    return create_op<2>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 3
    if (n_components == 3)
    return create_op<3>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 4
    if (n_components == 4)
    return create_op<4>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 5
    if (n_components == 5)
    return create_op<5>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 6
    if (n_components == 6)
    return create_op<6>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 7
    if (n_components == 7)
    return create_op<7>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 8
    if (n_components == 8)
    return create_op<8>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 9
    if (n_components == 9)
    return create_op<9>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 10
    if (n_components == 10)
    return create_op<10>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 11
    if (n_components == 11)
    return create_op<11>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 12
    if (n_components == 12)
    return create_op<12>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 13
    if (n_components == 13)
    return create_op<13>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 14
    if (n_components == 14)
    return create_op<14>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 15
    if (n_components == 15)
    return create_op<15>(level, matrix_free, q_point_operator);
  else
#endif
#if MAX_N_COMPONENTS >= 16
    if (n_components == 16)
    return create_op<16>(level, matrix_free, q_point_operator);
#endif

  AssertThrow(false, ExcNotImplemented());

  return create_op<1>(level, matrix_free, q_point_operator);
}

template <int dim, typename Number, typename VectorizedArrayType>
void
test(const Parameters &params, ConvergenceTable &table)
{
  using VectorType      = LinearAlgebra::distributed::Vector<Number>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<Number>;

  const auto fe_type              = params.fe_type;
  const auto fe_degree            = params.fe_degree;
  const auto n_subdivisions       = params.n_subdivisions;
  const auto n_global_refinements = params.n_global_refinements;
  const auto print_l2_norm        = false;

  std::unique_ptr<FiniteElement<dim>> fe;
  std::unique_ptr<Quadrature<dim>>    quadrature;
  MappingQ1<dim>                      mapping_q1;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  if (fe_type == "FE_Q")
    {
      AssertThrow(n_subdivisions == 1, ExcInternalError());

      fe = std::make_unique<FE_Q<dim>>(fe_degree);

      const unsigned int n_quadrature_points = params.n_quadrature_points > 0 ?
                                                 params.n_quadrature_points :
                                                 (fe_degree + 1);

      quadrature = std::make_unique<QGauss<dim>>(n_quadrature_points);
    }
  else if (fe_type == "FE_Q_iso_Q1")
    {
      AssertThrow(fe_degree == 1, ExcInternalError());

      fe = std::make_unique<FE_Q_iso_Q1<dim>>(n_subdivisions);
      quadrature =
        std::make_unique<QIterated<dim>>(QGauss<1>(2), n_subdivisions);
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

  MappingQCache<dim> mapping(1);


  mapping.initialize(
    mapping_q1,
    tria,
    [](const auto &, const auto &point) {
      Point<dim> result;

      if (true) // TODO
        return result;

      for (unsigned int d = 0; d < dim; ++d)
        result[d] = std::sin(2 * numbers::PI * point[(d + 1) % dim]) *
                    std::sin(numbers::PI * point[d]) * 0.01;

      return result;
    },
    true);

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;
  additional_data.overlap_communication_computation = false;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, *quadrature, additional_data);

  const auto run = [&](const auto fu) {
    // warm up
    for (unsigned int i = 0; i < 10; ++i)
      fu();

#ifdef LIKWID_PERFMON
    const auto add_padding = [](const int value) -> std::string {
      if (value < 10)
        return "000" + std::to_string(value);
      if (value < 100)
        return "00" + std::to_string(value);
      if (value < 1000)
        return "0" + std::to_string(value);
      if (value < 10000)
        return "" + std::to_string(value);

      AssertThrow(false, ExcInternalError());

      return "";
    };

    const std::string likwid_label =
      "likwid_" + add_padding(likwid_counter); // TODO
    likwid_counter++;
#endif

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START(likwid_label.c_str());
#endif

    const auto timer = std::chrono::system_clock::now();

    for (unsigned int i = 0; i < params.n_repetitions; ++i)
      fu();

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP(likwid_label.c_str());
#endif

    const double time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now() - timer)
                          .count() /
                        1e9;

    return time;
  };

  for (unsigned int n_components = 1;
       n_components <= (MAX_SINTERING_GRAINS + 2);
       ++n_components)
    {
      table.add_value("dim", dim);
      table.add_value("fe_type", fe_type);
      table.add_value("fe_degree", fe_degree);
      table.add_value("n_quadrature_points",
                      quadrature->get_tensor_basis()[0].size());
      table.add_value("n_subdivisions", n_subdivisions);
      table.add_value("n_global_refinements", n_global_refinements);
      table.add_value("n_repetitions", params.n_repetitions);
      table.add_value("n_dofs", dof_handler.n_dofs());
      table.add_value("n_components", n_components);


      HelmholtzQOperator q_point_operator_h;

      const double A       = 16;
      const double B       = 1;
      const double kappa_c = 1;
      const double kappa_p = 0.5;
      const double Mvol    = 1e-2;
      const double Mvap    = 1e-10;
      const double Msurf   = 4;
      const double Mgb     = 0.4;
      const double L       = 1;
      const double t       = 0.0;
      const double dt      = 0.1;

      const std::shared_ptr<MobilityProvider> mobility_provider =
        std::make_shared<ProviderAbstract>(Mvol, Mvap, Msurf, Mgb, L);

      FreeEnergy<VectorizedArrayType> free_energy(A, B);

      TimeIntegratorData<Number> time_data(
        std::make_unique<BDF1Scheme<Number>>(), dt);

      SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
        kappa_c, kappa_p, mobility_provider, std::move(time_data));

      sintering_data.set_n_components(n_components);
      sintering_data.set_time(t);

      AlignedVector<VectorizedArrayType> buffer;
      SinteringOperatorGenericResidualQuad<dim, VectorizedArrayType, 1>
        q_point_operator_s(free_energy, sintering_data, buffer);

      const auto run_op = [&](auto &q_point_operator, const std::string label) {
        // version 2: vectorial (block system)
        const auto projection_operator =
          create_op(n_components, params.level, matrix_free, q_point_operator);

        BlockVectorType src, dst;
        projection_operator->initialize_dof_vector(src);
        projection_operator->initialize_dof_vector(dst);

        for (unsigned int i = 0; i < n_components; ++i)
          VectorTools::interpolate(dof_handler,
                                   RightHandSide<dim>(i),
                                   src.block(i));

        unsigned int counter = 0;

        const auto time = run([&]() {
          projection_operator->vmult(dst, src);

          if (print_l2_norm && (counter++ == 0))
            pcout << dst.l2_norm() << std::endl;
        });

        table.add_value("t_vector_" + label, time / n_components);
        table.set_scientific("t_vector_" + label, true);
      };

      run_op(q_point_operator_h, "h");
      run_op(q_point_operator_s, "s");
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  AssertThrow(argc >= 2, ExcInternalError());

  ConvergenceTable table;

  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;

  for (int i = 1; i < argc; ++i)
    {
      Parameters params;
      params.parse(std::string(argv[i]));

      if (params.dim == 2)
        test<2, Number, VectorizedArrayType>(params, table);
      else if (params.dim == 3)
        test<3, Number, VectorizedArrayType>(params, table);
      else
        AssertThrow(false, ExcNotImplemented());
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
