#pragma once

#include <pf-applications/sintering/advection.h>
#include <pf-applications/sintering/operator_sintering_base.h>

namespace Sintering
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType>
  class SinteringOperatorGeneric
    : public SinteringOperatorBase<
        dim,
        Number,
        VectorizedArrayType,
        SinteringOperatorGeneric<dim, Number, VectorizedArrayType>>
  {
  public:
    using T = SinteringOperatorGeneric<dim, Number, VectorizedArrayType>;

    using VectorType = LinearAlgebra::distributed::Vector<Number>;
    using BlockVectorType =
      LinearAlgebra::distributed::DynamicBlockVector<Number>;

    using value_type  = Number;
    using vector_type = VectorType;

    SinteringOperatorGeneric(
      const MatrixFree<dim, Number, VectorizedArrayType> &        matrix_free,
      const AffineConstraints<Number> &                           constraints,
      const SinteringOperatorData<dim, VectorizedArrayType> &     data,
      const TimeIntegration::SolutionHistory<BlockVectorType> &   history,
      const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection,
      const bool                                                  matrix_based)
      : SinteringOperatorBase<
          dim,
          Number,
          VectorizedArrayType,
          SinteringOperatorGeneric<dim, Number, VectorizedArrayType>>(
          matrix_free,
          constraints,
          data,
          history,
          matrix_based)
      , advection(advection)
    {}

    ~SinteringOperatorGeneric()
    {}

    template <unsigned int with_time_derivative = 2>
    void
    evaluate_nonlinear_residual(BlockVectorType &      dst,
                                const BlockVectorType &src) const
    {
      MyScope scope(this->timer,
                    "sintering_op::nonlinear_residual",
                    this->do_timing);

#define OPERATION(c, d)                                           \
  MyMatrixFreeTools::cell_loop_wrapper(                           \
    this->matrix_free,                                            \
    &SinteringOperatorGeneric::                                   \
      do_evaluate_nonlinear_residual<c, d, with_time_derivative>, \
    this,                                                         \
    dst,                                                          \
    src,                                                          \
    true);
      EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
    }

    unsigned int
    n_components() const override
    {
      return this->data.n_components();
    }

    unsigned int
    n_grains() const
    {
      return this->n_components() - 2;
    }

    static constexpr unsigned int
    n_grains_to_n_components(const unsigned int n_grains)
    {
      return n_grains + 2;
    }

    template <int n_comp, int n_grains, typename FECellIntegratorType>
    void
    do_vmult_kernel(FECellIntegratorType &phi) const
    {
      AssertDimension(n_comp - 2, n_grains);

      const unsigned int cell = phi.get_current_cell_index();

      const auto &nonlinear_values    = this->data.get_nonlinear_values();
      const auto &nonlinear_gradients = this->data.get_nonlinear_gradients();

      const auto &free_energy = this->data.free_energy;
      const auto &mobility    = this->data.get_mobility();
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto  weight      = this->data.time_data.get_primary_weight();
      const auto &L           = mobility.Lgb();

      // Reinit advection data for the current cells batch
      if (this->advection.enabled())
        this->advection.reinit(cell,
                               static_cast<unsigned int>(n_grains),
                               phi.get_matrix_free());

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          typename FECellIntegratorType::value_type    value_result;
          typename FECellIntegratorType::gradient_type gradient_result;

          const auto  value        = phi.get_value(q);
          const auto  gradient     = phi.get_gradient(q);
          const auto &lin_value    = nonlinear_values[cell][q];
          const auto &lin_gradient = nonlinear_gradients[cell][q];

          const auto &lin_c_value = lin_value[0];

          const VectorizedArrayType *lin_etas_value = &lin_value[2];

          const auto lin_etas_value_power_2_sum =
            PowerHelper<n_grains, 2>::power_sum(lin_etas_value);



          // 1) process c row
          value_result[0] = value[0] * weight;

          gradient_result[0] = mobility.apply_M_derivative(
            &lin_value[0], &lin_gradient[0], n_grains, &value[0], &gradient[0]);



          // 2) process mu row
          value_result[1] =
            -value[1] +
            free_energy.d2f_dc2(lin_c_value, lin_etas_value) * value[0];

          for (unsigned int ig = 0; ig < n_grains; ++ig)
            value_result[1] +=
              free_energy.d2f_dcdetai(lin_c_value, lin_etas_value, ig) *
              value[ig + 2];

          gradient_result[1] = kappa_c * gradient[0];



          // 3) process eta rows
          for (unsigned int ig = 0; ig < n_grains; ++ig)
            {
              value_result[ig + 2] +=
                value[ig + 2] * weight +
                L * free_energy.d2f_dcdetai(lin_c_value, lin_etas_value, ig) *
                  value[0] +
                L *
                  free_energy.d2f_detai2(lin_c_value,
                                         lin_etas_value,
                                         lin_etas_value_power_2_sum,
                                         ig) *
                  value[ig + 2];

              gradient_result[ig + 2] = L * kappa_p * gradient[ig + 2];

              for (unsigned int jg = 0; jg < ig; ++jg)
                {
                  const auto d2f_detaidetaj = free_energy.d2f_detaidetaj(
                    lin_c_value, lin_etas_value, ig, jg);

                  value_result[ig + 2] += (L * d2f_detaidetaj) * value[jg + 2];
                  value_result[jg + 2] += (L * d2f_detaidetaj) * value[ig + 2];
                }
            }



          // 4) add advection contributations -> influences c AND etas
          if (this->advection.enabled())
            for (unsigned int ig = 0; ig < n_grains; ++ig)
              if (this->advection.has_velocity(ig))
                {
                  const auto &velocity_ig =
                    this->advection.get_velocity(ig, phi.quadrature_point(q));

                  value_result[0] += velocity_ig * gradient[0];

                  value_result[ig + 2] += velocity_ig * gradient[ig + 2];
                }



          phi.submit_value(value_result, q);
          phi.submit_gradient(gradient_result, q);
        }
    }

  private:
    template <int n_comp, int n_grains, int with_time_derivative>
    void
    do_evaluate_nonlinear_residual(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      BlockVectorType &                                   dst,
      const BlockVectorType &                             src,
      const std::pair<unsigned int, unsigned int> &       range) const
    {
      AssertDimension(n_comp - 2, n_grains);
      FECellIntegrator<dim, n_comp, Number, VectorizedArrayType> phi(
        matrix_free, this->dof_index);

      auto time_phi = this->time_integrator.create_cell_intergator(phi);

      const auto &free_energy = this->data.free_energy;
      const auto &mobility    = this->data.get_mobility();
      const auto &kappa_c     = this->data.kappa_c;
      const auto &kappa_p     = this->data.kappa_p;
      const auto &order       = this->data.time_data.get_order();
      const auto  weight      = this->data.time_data.get_primary_weight();
      const auto &L           = mobility.Lgb();

      const auto old_solutions = this->history.get_old_solutions();

      for (auto cell = range.first; cell < range.second; ++cell)
        {
          const auto &component_table = this->data.get_component_table()[cell];

          phi.reinit(cell);
          phi.gather_evaluate(src,
                              EvaluationFlags::EvaluationFlags::values |
                                EvaluationFlags::EvaluationFlags::gradients);

          if (with_time_derivative == 2)
            for (unsigned int i = 0; i < order; ++i)
              {
                time_phi[i].reinit(cell);
                time_phi[i].read_dof_values_plain(*old_solutions[i]);
                time_phi[i].evaluate(EvaluationFlags::EvaluationFlags::values);
              }

          // Reinit advection data for the current cells batch
          if (this->advection.enabled())
            this->advection.reinit(cell,
                                   static_cast<unsigned int>(n_grains),
                                   matrix_free);

          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              auto value    = phi.get_value(q);
              auto gradient = phi.get_gradient(q);

              for (unsigned int ig = 0; ig < n_grains; ++ig)
                if (component_table[ig] == false)
                  {
                    value[ig + 2]    = VectorizedArrayType();
                    gradient[ig + 2] = Tensor<1, dim, VectorizedArrayType>();
                  }

              const VectorizedArrayType *                etas_value = &value[2];
              const Tensor<1, dim, VectorizedArrayType> *etas_gradient =
                &gradient[2];

              const auto etas_value_power_2_sum =
                PowerHelper<n_grains, 2>::power_sum(etas_value);
              const auto etas_value_power_3_sum =
                PowerHelper<n_grains, 3>::power_sum(etas_value);

              Tensor<1, n_comp, VectorizedArrayType> value_result;
              Tensor<1, n_comp, Tensor<1, dim, VectorizedArrayType>>
                gradient_result;



              // 1) process c row
              if (with_time_derivative == 2)
                this->time_integrator.compute_time_derivative(
                  value_result[0], value, time_phi, 0, q);
              else if (with_time_derivative == 1)
                value_result[0] = value[0] * weight;

              gradient_result[0] = mobility.apply_M(value[0],
                                                    etas_value,
                                                    n_grains,
                                                    gradient[0],
                                                    etas_gradient,
                                                    gradient[1]);



              // 2) process mu row
              value_result[1] =
                -value[1] + free_energy.df_dc(value[0],
                                              etas_value,
                                              etas_value_power_2_sum,
                                              etas_value_power_3_sum);
              gradient_result[1] = kappa_c * gradient[0];



              // 3) process eta rows
              for (unsigned int ig = 0; ig < n_grains; ++ig)
                {
                  value_result[2 + ig] =
                    L * free_energy.df_detai(value[0],
                                             etas_value,
                                             etas_value_power_2_sum,
                                             ig);

                  if (with_time_derivative == 2)
                    this->time_integrator.compute_time_derivative(
                      value_result[2 + ig], value, time_phi, 2 + ig, q);
                  else if (with_time_derivative == 1)
                    value_result[ig + 2] += value[ig + 2] * weight;

                  gradient_result[2 + ig] = L * kappa_p * gradient[2 + ig];
                }



              // 4) add advection contributations -> influences c AND etas
              if (this->advection.enabled())
                for (unsigned int ig = 0; ig < n_grains; ++ig)
                  if (this->advection.has_velocity(ig))
                    {
                      const auto &velocity_ig =
                        this->advection.get_velocity(ig,
                                                     phi.quadrature_point(q));

                      value_result[0] += velocity_ig * gradient[0];
                      value_result[2 + ig] += velocity_ig * gradient[2 + ig];
                    }


              for (unsigned int ig = 0; ig < n_grains; ++ig)
                if (component_table[ig] == false)
                  {
                    value_result[ig + 2] = VectorizedArrayType();
                    gradient_result[ig + 2] =
                      Tensor<1, dim, VectorizedArrayType>();
                  }

              phi.submit_value(value_result, q);
              phi.submit_gradient(gradient_result, q);
            }
          phi.integrate_scatter(EvaluationFlags::EvaluationFlags::values |
                                  EvaluationFlags::EvaluationFlags::gradients,
                                dst);
        }
    }

    const AdvectionMechanism<dim, Number, VectorizedArrayType> &advection;
  };
} // namespace Sintering
