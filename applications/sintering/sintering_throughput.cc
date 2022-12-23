// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

#define USE_FE_Q_iso_Q1

#ifdef USE_FE_Q_iso_Q1
#  define FE_DEGREE 2
#  define N_Q_POINTS_1D FE_DEGREE * 2
#else
#  define FE_DEGREE 1
#  define N_Q_POINTS_1D FE_DEGREE + 1
#endif

#define WITH_TIMING
//#define WITH_TIMING_OUTPUT

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/revision.h>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

#include <pf-applications/base/revision.h>

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/mobility.h>
#include <pf-applications/sintering/operator_sintering_generic.h>
#include <pf-applications/sintering/preconditioners.h>
#include <pf-applications/sintering/sintering_data.h>

using namespace dealii;
using namespace Sintering;

// helper function
double
run(const std::function<void()> &fu)
{
  const unsigned int n_repetitions = 100;

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

  static unsigned int likwid_counter = 0;

  const std::string likwid_label =
    "likwid_" + add_padding(likwid_counter); // TODO
  likwid_counter++;
#endif

  MPI_Barrier(MPI_COMM_WORLD);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START(likwid_label.c_str());
#endif

  const auto timer = std::chrono::system_clock::now();

  for (unsigned int i = 0; i < n_repetitions; ++i)
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
}

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
  using Number                            = double;
  using VectorizedArrayType               = VectorizedArray<Number>;
  using VectorType = LinearAlgebra::distributed::DynamicBlockVector<Number>;

  // some arbitrary constants
  const double        A                      = 16;
  const double        B                      = 1;
  const double        kappa_c                = 1;
  const double        kappa_p                = 0.5;
  const double        Mvol                   = 1e-2;
  const double        Mvap                   = 1e-10;
  const double        Msurf                  = 4;
  const double        Mgb                    = 0.4;
  const double        L                      = 1;
  const double        time_integration_order = 1;
  const double        t                      = 0.0;
  const double        dt                     = 0.1;
  std::vector<double> dts(time_integration_order, 0.0);
  dts[0] = dt;

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
  additional_data.overlap_communication_computation = false;

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, *quadrature, additional_data);

  ConvergenceTable table;

  for (unsigned int n_grains = 2; n_grains <= MAX_SINTERING_GRAINS; ++n_grains)
    {
      const unsigned int n_components = n_grains + 2;

      table.add_value("n_components", n_components);

      if (true) // test Helmholtz operator
        {
          HelmholtzOperator<dim, Number, VectorizedArrayType>
            helmholtz_operator(matrix_free, constraints, n_components);

          VectorType src, dst;
          helmholtz_operator.initialize_dof_vector(src);
          helmholtz_operator.initialize_dof_vector(dst);
          src = 1.0;

          const auto time = run([&]() { helmholtz_operator.vmult(dst, src); });

          table.add_value("t_helmholtz", time);
          table.set_scientific("t_helmholtz", true);
        }

      if (true) // test sintering operator
        {
          const std::shared_ptr<MobilityProvider> mobility_provider =
            std::make_shared<ProviderAbstract>(Mvol, Mvap, Msurf, Mgb, L);

          TimeIntegration::SolutionHistory<VectorType> solution_history(
            time_integration_order + 1);

          SinteringOperatorData<dim, VectorizedArrayType> sintering_data(
            A, B, kappa_c, kappa_p, mobility_provider, time_integration_order);

          sintering_data.set_n_components(n_components);
          sintering_data.time_data.set_all_dt(dts);
          sintering_data.set_time(t);

          AdvectionMechanism<dim, Number, VectorizedArrayType> advection;

          SinteringOperatorGeneric<dim, Number, VectorizedArrayType>
            nonlinear_operator(matrix_free,
                               constraints,
                               sintering_data,
                               solution_history,
                               advection,
                               false);

          VectorType src, dst;
          nonlinear_operator.initialize_dof_vector(src);
          nonlinear_operator.initialize_dof_vector(dst);
          src = 1.0;

          sintering_data.fill_quadrature_point_values(matrix_free,
                                                      src,
                                                      false,
                                                      false);

          const auto time = run([&]() { nonlinear_operator.vmult(dst, src); });

          table.add_value("t_sintering", time);
          table.set_scientific("t_sintering", true);
        }
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    table.write_text(std::cout, TableHandler::TextOutputFormat::org_mode_table);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
