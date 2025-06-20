// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2024 by the hpsint authors
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

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_idr.h>
#include <deal.II/lac/trilinos_solver.h>

#include <pf-applications/base/timer.h>

#include <pf-applications/lac/dynamic_block_vector.h>
#include <pf-applications/lac/solvers_linear_parameters.h>
#include <pf-applications/lac/solvers_nonlinear_parameters.h>

#include <pf-applications/numerics/vector_tools.h>

namespace LinearSolvers
{
  using namespace dealii;

  template <typename Number>
  class LinearSolverBase
  {
  public:
    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    virtual ~LinearSolverBase() = default;

    virtual unsigned int
    solve(VectorType &dst, const VectorType &src) = 0;

    virtual unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) = 0;
  };



  template <typename Operator, typename Preconditioner>
  class SolverGMRESWrapper
    : public LinearSolverBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::vector_type;
    using BlockVectorType = typename Operator::BlockVectorType;

    SolverGMRESWrapper(const Operator       &op,
                       const Preconditioner &preconditioner,
                       SolverControl        &solver_control,
                       const GMRESData      &data = GMRESData())
      : op(op)
      , preconditioner(preconditioner)
      , solver_control(solver_control)
      , data(data)
    {}

    unsigned int
    solve(VectorType &dst, const VectorType &src) override
    {
      return solve_internal(dst, src);
    }

    unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) override
    {
      return solve_internal(dst, src);
    }

  private:
    template <typename T>
    unsigned int
    solve_internal(T &dst, const T &src)
    {
      MyScope scope(timer, "gmres::solve");

      typename SolverGMRES<T>::AdditionalData additional_data;

      if (data.orthogonalization_strategy == "classical gram schmidt")
        additional_data.orthogonalization_strategy =
          LinearAlgebra::OrthogonalizationStrategy::classical_gram_schmidt;
      else if (data.orthogonalization_strategy == "modified gram schmidt")
        additional_data.orthogonalization_strategy =
          LinearAlgebra::OrthogonalizationStrategy::modified_gram_schmidt;
      else
        AssertThrow(false, ExcNotImplemented());

      SolverGMRES<T> solver(solver_control, additional_data);
      solver.solve(op, dst, src, preconditioner);

      return solver_control.last_step();
    }

    const Operator       &op;
    const Preconditioner &preconditioner;
    SolverControl        &solver_control;
    const GMRESData       data;

    mutable MyTimerOutput timer;
  };



  template <typename Operator, typename Preconditioner>
  class SolverRelaxation
    : public LinearSolverBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::vector_type;
    using BlockVectorType = typename Operator::BlockVectorType;

    SolverRelaxation(const Operator       &op,
                     const Preconditioner &preconditioner,
                     const double          relaxation   = 1.,
                     const unsigned int    n_iterations = 1)
      : op(op)
      , preconditioner(preconditioner)
      , relaxation(relaxation)
      , n_iterations(n_iterations)
    {}

    unsigned int
    solve(VectorType &dst, const VectorType &src) override
    {
      return solve_internal(dst, src);
    }

    unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) override
    {
      return solve_internal(dst, src);
    }

  private:
    template <typename T>
    unsigned int
    solve_internal(T &dst, const T &src)
    {
      MyScope scope(timer, "relaxation::solve");

      typename PreconditionRelaxation<Operator,
                                      const Preconditioner>::AdditionalData
        additional_data;
      additional_data.relaxation   = relaxation;
      additional_data.n_iterations = n_iterations;
      additional_data.preconditioner =
        std::shared_ptr<const Preconditioner>(&preconditioner,
                                              [](const auto &) {});

      PreconditionRelaxation<Operator, const Preconditioner> solver;
      solver.initialize(op, additional_data);
      solver.vmult(dst, src);

      return n_iterations;
    }

    const Operator       &op;
    const Preconditioner &preconditioner;
    const double          relaxation;
    const unsigned int    n_iterations;

    mutable MyTimerOutput timer;
  };



  template <typename Operator, typename Preconditioner>
  class SolverIDRWrapper
    : public LinearSolverBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::vector_type;
    using BlockVectorType = typename Operator::BlockVectorType;

    SolverIDRWrapper(const Operator       &op,
                     const Preconditioner &preconditioner,
                     SolverControl        &solver_control)
      : op(op)
      , preconditioner(preconditioner)
      , solver_control(solver_control)
    {}

    unsigned int
    solve(VectorType &dst, const VectorType &src) override
    {
      return solve_internal(dst, src);
    }

    unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) override
    {
      return solve_internal(dst, src);
    }

  private:
    template <typename T>
    unsigned int
    solve_internal(T &dst, const T &src)
    {
      MyScope scope(timer, "idr::solve");

      SolverIDR<T> solver(solver_control);
      solver.solve(op, dst, src, preconditioner);

      return solver_control.last_step();
    }

    const Operator       &op;
    const Preconditioner &preconditioner;
    SolverControl        &solver_control;

    mutable MyTimerOutput timer;
  };



  template <typename Operator, typename Preconditioner>
  class SolverBicgstabWrapper
    : public LinearSolverBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::vector_type;
    using BlockVectorType = typename Operator::BlockVectorType;

    SolverBicgstabWrapper(const Operator       &op,
                          const Preconditioner &preconditioner,
                          SolverControl        &solver_control,
                          const unsigned int    max_bicgsteps)
      : op(op)
      , preconditioner(preconditioner)
      , solver_control(solver_control)
      , max_bicgsteps(max_bicgsteps)
    {}

    unsigned int
    solve(VectorType &dst, const VectorType &src) override
    {
      return solve_internal(dst, src);
    }

    unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) override
    {
      return solve_internal(dst, src);
    }

  private:
    template <typename T>
    unsigned int
    solve_internal(T &dst, const T &src)
    {
      MyScope scope(timer, "bicgstab::solve");

      // copy information to be able to restart with GMRESs
      const unsigned int max_steps = solver_control.max_steps();

      T dst_copy;
      dst_copy.reinit(dst);
      dst_copy.copy_locally_owned_data_from(dst);

      try
        {
          // try to solve with Bicgstab with a reasonable number
          // of iterations
          solver_control.set_max_steps(max_bicgsteps);

          SolverBicgstab<T> solver(solver_control);
          solver.solve(op, dst, src, preconditioner);

          solver_control.set_max_steps(max_steps);

          return solver_control.last_step();
        }
      catch (const SolverControl::NoConvergence &)
        {
          // reset data
          solver_control.set_max_steps(max_steps);
          dst.copy_locally_owned_data_from(dst_copy);

          // solve with GMRES
          SolverGMRES<T> solver(solver_control);
          solver.solve(op, dst, src, preconditioner);

          return solver_control.last_step() + max_bicgsteps;
        }
    }

    const Operator       &op;
    const Preconditioner &preconditioner;
    SolverControl        &solver_control;
    const unsigned int    max_bicgsteps;

    mutable MyTimerOutput timer;
  };

  template <typename Operator>
  class SolverDirectWrapper
    : public LinearSolverBase<typename Operator::value_type>
  {
  public:
    using VectorType      = typename Operator::vector_type;
    using BlockVectorType = typename Operator::BlockVectorType;

    SolverDirectWrapper(const Operator &op, SolverControl &solver_control)
      : op(op)
      , solver_control(solver_control)
      , solver(solver_control)
    {}

    unsigned int
    solve(VectorType &dst, const VectorType &src) override
    {
      return solve_internal(dst, src);
    }

    unsigned int
    solve(BlockVectorType &dst, const BlockVectorType &src) override
    {
      VectorType src_, dst_;

      const auto partitioner = op.get_system_partitioner();

      src_.reinit(partitioner);
      dst_.reinit(partitioner);

      VectorTools::merge_components_fast(src, src_);
      auto ret_val = solve_internal(dst_, src_);
      VectorTools::split_up_components_fast(dst_, dst);

      return ret_val;
    }

  private:
    template <typename T>
    unsigned int
    solve_internal(T &dst, const T &src)
    {
      MyScope scope(timer, "direct::solve");

      const auto &mtr = op.get_system_matrix();

      solver.initialize(mtr);
      solver.solve(dst, src);

      return solver_control.last_step();
    }

    const Operator                &op;
    SolverControl                 &solver_control;
    TrilinosWrappers::SolverDirect solver;

    mutable MyTimerOutput timer;
  };

  template <typename JacobianOperator, typename Preconditioner>
  std::unique_ptr<LinearSolverBase<typename JacobianOperator::value_type>>
  create(const JacobianOperator                &jacobian_operator,
         const Preconditioner                  &preconditioner,
         SolverControl                         &solver_control_l,
         const NonLinearSolvers::NonLinearData &params,
         const std::string                     &label)
  {
    if (label == "GMRES")
      return std::make_unique<
        LinearSolvers::SolverGMRESWrapper<JacobianOperator, Preconditioner>>(
        jacobian_operator, preconditioner, solver_control_l, params.gmres_data);
    else if (label == "Relaxation")
      return std::make_unique<
        LinearSolvers::SolverRelaxation<JacobianOperator, Preconditioner>>(
        jacobian_operator, preconditioner);
    else if (label == "IDR")
      return std::make_unique<
        LinearSolvers::SolverIDRWrapper<JacobianOperator, Preconditioner>>(
        jacobian_operator, preconditioner, solver_control_l);
    else if (label == "Bicgstab")
      return std::make_unique<
        LinearSolvers::SolverBicgstabWrapper<JacobianOperator, Preconditioner>>(
        jacobian_operator,
        preconditioner,
        solver_control_l,
        params.l_bisgstab_tries);

    AssertThrow(false,
                ExcMessage("Linear solver << " + label + " >> not known!"));

    return {};
  }

  template <typename JacobianOperator,
            typename Preconditioner,
            typename NonLinearOperator>
  std::unique_ptr<LinearSolverBase<typename JacobianOperator::value_type>>
  create(const JacobianOperator                &jacobian_operator,
         const Preconditioner                  &preconditioner,
         const NonLinearOperator               &nonlinear_operator,
         SolverControl                         &solver_control_l,
         const NonLinearSolvers::NonLinearData &params,
         const std::string                     &label)
  {
    if (label == "Direct")
      return std::make_unique<
        LinearSolvers::SolverDirectWrapper<NonLinearOperator>>(
        nonlinear_operator, solver_control_l);
    else
      return create(
        jacobian_operator, preconditioner, solver_control_l, params, label);
  }
} // namespace LinearSolvers
