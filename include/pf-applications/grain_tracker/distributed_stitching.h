// ---------------------------------------------------------------------
//
// Copyright (C) 2023-2025 by the hpsint authors
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

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/data_out.h>

#include <pf-applications/base/scoped_name.h>
#include <pf-applications/base/timer.h>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include <pf-applications/grain_tracker/motion.h>

namespace GrainTracker
{
  using namespace dealii;

  template <int dim, typename VectorSolution, typename VectorIds>
  unsigned int
  run_flooding(const typename DoFHandler<dim>::cell_iterator &cell,
               const VectorSolution                          &solution,
               VectorIds                                     &particle_ids,
               const unsigned int                             id,
               double                                        &max_value,
               const double threshold_lower     = 0,
               const double invalid_particle_id = -1.0)
  {
    if (cell->has_children())
      {
        unsigned int counter = 0;

        for (const auto &child : cell->child_iterators())
          counter += run_flooding<dim>(child,
                                       solution,
                                       particle_ids,
                                       id,
                                       max_value,
                                       threshold_lower,
                                       invalid_particle_id);

        return counter;
      }

    if (cell->is_locally_owned() == false)
      return 0;

    const auto particle_id = particle_ids[cell->global_active_cell_index()];

    if (particle_id != invalid_particle_id)
      return 0; // cell has been visited

    Vector<double> values(cell->get_fe().n_dofs_per_cell());

    cell->get_dof_values(solution, values);

    const auto cell_max_value = *std::max_element(values.begin(), values.end());
    const bool has_particle   = cell_max_value > threshold_lower;

    if (!has_particle)
      return 0; // cell has no particle

    particle_ids[cell->global_active_cell_index()] = id;

    max_value = std::max(max_value, cell_max_value);

    unsigned int counter = 1;

    for (const auto face : cell->face_indices())
      if (cell->at_boundary(face) == false)
        counter += run_flooding<dim>(cell->neighbor(face),
                                     solution,
                                     particle_ids,
                                     id,
                                     max_value,
                                     threshold_lower,
                                     invalid_particle_id);

    return counter;
  }

  std::vector<unsigned int>
  connected_components(
    const unsigned int                                         N,
    const std::vector<std::tuple<unsigned int, unsigned int>> &edges);

  std::vector<unsigned int>
  perform_distributed_stitching_via_graph(
    const MPI_Comm comm,
    const std::vector<std::vector<std::tuple<unsigned int, unsigned int>>>
                  &edges_in,
    MyTimerOutput *timer = nullptr);

  std::vector<unsigned int>
  perform_distributed_stitching(
    const MPI_Comm                                                   comm,
    std::vector<std::vector<std::tuple<unsigned int, unsigned int>>> input,
    MyTimerOutput *timer = nullptr);

  template <int dim, typename VectorIds>
  auto
  build_local_connectivity(const DoFHandler<dim> &dof_handler,
                           const VectorIds       &particle_ids,
                           const double           local_grains_num,
                           const double           local_offset,
                           const double           invalid_particle_id = -1.0)
  {
    std::vector<std::vector<std::tuple<unsigned int, unsigned int>>>
      local_connectivity(local_grains_num);

    for (const auto &ghost_cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (ghost_cell->is_ghost())
        {
          const auto particle_id =
            particle_ids[ghost_cell->global_active_cell_index()];

          if (particle_id == invalid_particle_id)
            continue;

          for (const auto face : ghost_cell->face_indices())
            {
              if (ghost_cell->at_boundary(face))
                continue;

              const auto add = [&](const auto &ghost_cell,
                                   const auto &local_cell) {
                if (local_cell->is_locally_owned() == false)
                  return;

                const auto neighbor_particle_id =
                  particle_ids[local_cell->global_active_cell_index()];

                if (neighbor_particle_id == invalid_particle_id)
                  return;

                auto &temp =
                  local_connectivity[neighbor_particle_id - local_offset];
                temp.emplace_back(ghost_cell->subdomain_id(), particle_id);
                std::sort(temp.begin(), temp.end());
                temp.erase(std::unique(temp.begin(), temp.end()), temp.end());
              };

              if (ghost_cell->neighbor(face)->has_children())
                {
                  for (unsigned int subface = 0;
                       subface <
                       GeometryInfo<dim>::n_subfaces(
                         dealii::internal::SubfaceCase<dim>::case_isotropic);
                       ++subface)
                    add(ghost_cell,
                        ghost_cell->neighbor_child_on_subface(face, subface));
                }
              else
                add(ghost_cell, ghost_cell->neighbor(face));
            }
        }

    return local_connectivity;
  }

  unsigned int
  number_of_stitched_particles(
    const std::vector<unsigned int> &local_to_global_particle_ids,
    const MPI_Comm                   comm);

  template <typename VectorIds>
  void
  switch_to_global_indices(
    VectorIds                       &particle_ids,
    const std::vector<unsigned int> &local_to_global_particle_ids,
    const unsigned int               offset,
    const double                     invalid_particle_id = -1.0)
  {
    const unsigned int n_local_particles = local_to_global_particle_ids.size();
    (void)n_local_particles;

    for (auto &particle_id : particle_ids)
      if (particle_id != invalid_particle_id)
        {
          const unsigned int local_id =
            static_cast<unsigned int>(particle_id) - offset;

          AssertIndexRange(local_id, n_local_particles);

          particle_id = local_to_global_particle_ids[local_id];
        }

    particle_ids.update_ghost_values();
  }

  template <int dim, typename VectorIds>
  std::vector<double>
  compute_particles_max_values(
    const DoFHandler<dim>           &dof_handler,
    const VectorIds                 &particle_ids,
    const std::vector<unsigned int> &local_to_global_particle_ids,
    const unsigned int               local_offset,
    const double                     invalid_particle_id       = -1.0,
    const std::vector<double>       &local_particle_max_values = {})
  {
    const auto comm = dof_handler.get_communicator();

    const unsigned int n_particles =
      number_of_stitched_particles(local_to_global_particle_ids, comm);

    std::vector<double> particle_max_values(n_particles);

    // Compute local information
    for (const auto &cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto particle_id =
            particle_ids[cell->global_active_cell_index()];

          if (particle_id == invalid_particle_id)
            continue;

          const unsigned int local_id =
            static_cast<unsigned int>(particle_id) - local_offset;
          const unsigned int unique_id = local_to_global_particle_ids[local_id];

          AssertIndexRange(unique_id, n_particles);

          particle_max_values[unique_id] = local_particle_max_values[local_id];
        }

    // Reduce information - particles max values
    MPI_Allreduce(MPI_IN_PLACE,
                  particle_max_values.data(),
                  particle_max_values.size(),
                  MPI_DOUBLE,
                  MPI_MAX,
                  comm);

    return particle_max_values;
  }

  template <int dim, typename VectorIds>
  std::tuple<std::vector<Point<dim>>, // particle_centers
             std::vector<double>>     // particle_measures
  compute_particles_info(const DoFHandler<dim> &dof_handler,
                         const VectorIds       &particle_ids,
                         const unsigned int     n_particles,
                         const double           invalid_particle_id = -1.0)
  {
    const auto comm = dof_handler.get_communicator();

    const unsigned int  n_features = 1 + dim;
    std::vector<double> particle_info(n_particles * n_features);
    std::vector<double> particle_max_values(n_particles);

    // Compute local information
    for (const auto &cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto unique_id = particle_ids[cell->global_active_cell_index()];

          if (unique_id == invalid_particle_id)
            continue;

          AssertIndexRange(unique_id, n_particles);

          particle_info[n_features * unique_id + 0] += cell->measure();

          for (unsigned int d = 0; d < dim; ++d)
            particle_info[n_features * unique_id + 1 + d] +=
              cell->center()[d] * cell->measure();
        }

    // Reduce information - particles info
    MPI_Allreduce(MPI_IN_PLACE,
                  particle_info.data(),
                  particle_info.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);

    // Compute particles centers
    std::vector<Point<dim>> particle_centers(n_particles);
    std::vector<double>     particle_measures(n_particles);
    for (unsigned int i = 0; i < n_particles; i++)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            particle_centers[i][d] = particle_info[i * n_features + 1 + d] /
                                     particle_info[i * n_features];
          }
        particle_measures[i] = particle_info[i * n_features];
      }

    return std::make_tuple(std::move(particle_centers),
                           std::move(particle_measures));
  }

  template <int dim, typename VectorIds>
  std::tuple<std::vector<double>,     // particle_radii
             std::vector<Point<dim>>> // particle_remotes
  compute_particles_radii(const DoFHandler<dim>         &dof_handler,
                          const VectorIds               &particle_ids,
                          const std::vector<Point<dim>> &particle_centers,
                          const bool   evaluate_remotes    = false,
                          const double invalid_particle_id = -1.0)
  {
    const auto comm = dof_handler.get_communicator();

    const unsigned int n_particles = particle_centers.size();

    // Compute particles radii
    std::vector<double>     particle_radii(n_particles, 0.);
    std::vector<Point<dim>> particle_remotes(evaluate_remotes ? n_particles :
                                                                0);
    for (const auto &cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto unique_id = particle_ids[cell->global_active_cell_index()];

          if (unique_id == invalid_particle_id)
            continue;

          AssertIndexRange(unique_id, n_particles);

          const auto &center = particle_centers[unique_id];

          auto dist_vec = cell->barycenter() - center;
          dist_vec += (dist_vec / dist_vec.norm()) * cell->diameter() / 2.;

          const double dist =
            center.distance(cell->barycenter()) + cell->diameter() / 2.;

          if (evaluate_remotes && dist > particle_radii[unique_id])
            particle_remotes[unique_id] = dist_vec;

          particle_radii[unique_id] = std::max(particle_radii[unique_id], dist);
        }

    std::vector<double> particle_radii_local(particle_radii);

    // Reduce information - particles radii
    MPI_Allreduce(MPI_IN_PLACE,
                  particle_radii.data(),
                  particle_radii.size(),
                  MPI_DOUBLE,
                  MPI_MAX,
                  comm);

    // Exchange the remote points
    if (evaluate_remotes)
      {
        // If the current rank is not the owner of the furthest point, then we
        // nullify it since we perform a global summation later.
        for (unsigned int unique_id = 0; unique_id < particle_radii.size();
             ++unique_id)
          if (std::abs(particle_radii[unique_id] -
                       particle_radii_local[unique_id]) > 1e-16)
            particle_remotes[unique_id] = Point<dim>();

        // Perform global communication
        MPI_Allreduce(MPI_IN_PLACE,
                      particle_remotes.begin()->begin_raw(),
                      particle_remotes.size() * dim,
                      MPI_DOUBLE,
                      MPI_SUM,
                      comm);
      }

    return std::make_tuple(std::move(particle_radii),
                           std::move(particle_remotes));
  }

  template <int dim, typename VectorIds>
  std::vector<double>
  compute_particles_inertia(const DoFHandler<dim>         &dof_handler,
                            const VectorIds               &particle_ids,
                            const std::vector<Point<dim>> &particle_centers,
                            const double invalid_particle_id = -1.0)
  {
    const auto comm = dof_handler.get_communicator();

    const unsigned int n_particles = particle_centers.size();

    // Compute particles moments of inertia
    std::vector<double> particle_inertia(n_particles * num_inertias<dim>, 0.);
    for (const auto &cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto unique_id = particle_ids[cell->global_active_cell_index()];

          if (unique_id == invalid_particle_id)
            continue;

          AssertIndexRange(unique_id, n_particles);

          const auto &center  = particle_centers[unique_id];
          const auto  r_local = Point<dim>(cell->center() - center);

          evaluate_inertia_properties(
            r_local,
            cell->measure(),
            &(particle_inertia[num_inertias<dim> * unique_id]));
        }

    // Reduce information - particles info
    MPI_Allreduce(MPI_IN_PLACE,
                  particle_inertia.data(),
                  particle_inertia.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);

    return particle_inertia;
  }

  template <int dim, typename VectorSolution, typename VectorIds>
  std::tuple<unsigned int, std::vector<unsigned int>, std::vector<double>>
  detect_local_particle_groups(VectorIds             &particle_ids,
                               const DoFHandler<dim> &dof_handler,
                               const VectorSolution  &solution,
                               const bool     stitching_via_graphs = true,
                               const double   threshold_lower      = 0.01,
                               const double   invalid_particle_id  = -1.0,
                               MyTimerOutput *timer                = nullptr)
  {
    const MPI_Comm comm = dof_handler.get_communicator();

    // step 1) run flooding and determine local particles and give them
    // local ids
    particle_ids = invalid_particle_id;

    unsigned int counter      = 0;
    unsigned int offset       = 0;
    double       op_max_value = std::numeric_limits<double>::lowest();

    std::vector<double> local_particle_max_values;

    {
      ScopedName sc("run_flooding");
      MyScope    scope(sc, timer);

      const bool has_ghost_elements = solution.has_ghost_elements();

      if (has_ghost_elements == false)
        solution.update_ghost_values();

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (run_flooding<dim>(cell,
                                solution,
                                particle_ids,
                                counter,
                                op_max_value,
                                threshold_lower,
                                invalid_particle_id) > 0)
            {
              counter++;
              local_particle_max_values.push_back(op_max_value);
              op_max_value = std::numeric_limits<double>::lowest();
            }
        }

      if (has_ghost_elements == false)
        solution.zero_out_ghost_values();
    }

    // step 2) determine the global number of locally determined particles
    // and give each one an unique id by shifting the ids
    MPI_Exscan(&counter, &offset, 1, MPI_UNSIGNED, MPI_SUM, comm);

    for (auto &particle_id : particle_ids)
      if (particle_id != invalid_particle_id)
        particle_id += offset;

    // step 3) get particle ids on ghost cells and figure out if local
    // particles and ghost particles might be one particle
    particle_ids.update_ghost_values();

    auto local_connectivity = build_local_connectivity(
      dof_handler, particle_ids, counter, offset, invalid_particle_id);

    // step 4) based on the local-ghost information, figure out all
    // particles on all processes that belong togher (unification ->
    // clique), give each clique an unique id, and return mapping from the
    // global non-unique ids to the global ids
    std::vector<unsigned int> local_to_global_particle_ids;
    {
      ScopedName sc("distributed_stitching");
      MyScope    scope(sc, timer);

      local_to_global_particle_ids =
        stitching_via_graphs ?
          perform_distributed_stitching_via_graph(comm,
                                                  local_connectivity,
                                                  timer) :
          perform_distributed_stitching(comm, local_connectivity, timer);
    }

    return std::make_tuple(offset,
                           std::move(local_to_global_particle_ids),
                           std::move(local_particle_max_values));
  }

  namespace internal
  {
    /* This function searches for the top layer of cells constituting each of
     * the particle clique and add them to the agglomerations container. */
    template <int dim, typename VectorIds>
    void
    run_flooding_prep(
      const typename DoFHandler<dim>::cell_iterator &cell,
      const VectorIds                               &particle_ids,
      VectorIds                                     &particle_markers,
      std::deque<std::vector<DoFCellAccessor<dim, dim, false>>> &agglomerations,
      const double invalid_particle_id = -1.0)
    {
      if (cell->has_children())
        {
          for (const auto &child : cell->child_iterators())
            run_flooding_prep<dim>(child,
                                   particle_ids,
                                   particle_markers,
                                   agglomerations,
                                   invalid_particle_id);

          return;
        }

      if (!cell->is_locally_owned())
        return;

      const auto particle_id = particle_ids[cell->global_active_cell_index()];

      // If cell does not belong to any particle - skip it
      if (particle_id == invalid_particle_id)
        return;

      const auto particle_marker =
        particle_markers[cell->global_active_cell_index()];

      // If cell has been visited - skip it
      if (particle_marker != invalid_particle_id)
        return;

      // Use global particle ids for markers
      particle_markers[cell->global_active_cell_index()] = particle_id;

      for (const auto face : cell->face_indices())
        if (!cell->at_boundary(face))
          {
            const auto neighbor = cell->neighbor(face);

            if (!neighbor->has_children())
              {
                const auto neighbor_particle_id =
                  particle_ids[neighbor->global_active_cell_index()];

                if (neighbor_particle_id == invalid_particle_id)
                  agglomerations[agglomerations.size() - cell->level() - 1]
                    .push_back(*cell);
                else
                  run_flooding_prep<dim>(neighbor,
                                         particle_ids,
                                         particle_markers,
                                         agglomerations,
                                         invalid_particle_id);
              }
            else
              {
                for (const auto &child : neighbor->child_iterators())
                  {
                    if (!child->is_active() || !child->is_locally_owned())
                      continue;

                    const auto child_particle_id =
                      particle_ids[child->global_active_cell_index()];

                    if (child_particle_id == invalid_particle_id)
                      {
                        for (const auto child_face : child->face_indices())
                          if (!child->at_boundary(child_face))
                            {
                              const auto neighbor_of_child =
                                child->neighbor(child_face);

                              if (neighbor_of_child->is_active() &&
                                  neighbor_of_child->is_locally_owned() &&
                                  neighbor_of_child->id() == cell->id())
                                {
                                  agglomerations[agglomerations.size() -
                                                 cell->level() - 1]
                                    .push_back(*cell);
                                  break;
                                }
                            }
                      }
                  }

                run_flooding_prep<dim>(neighbor,
                                       particle_ids,
                                       particle_markers,
                                       agglomerations,
                                       invalid_particle_id);
              }
          }
    }
  } // namespace internal

  template <int dim, typename VectorIds>
  std::map<std::pair<unsigned int, unsigned int>, double>
  estimate_particle_distances(const VectorIds       &particle_ids,
                              const DoFHandler<dim> &dof_handler,
                              const double           invalid_particle_id = -1.0,
                              MyTimerOutput         *timer    = nullptr,
                              DataOut<dim>          *data_out = nullptr)
  {
    ScopedName sc("estimate_particle_distances");
    MyScope    scope(sc, timer);

    // Create other 2 vectors using the same partitioning as the input one
    VectorIds particle_distances(particle_ids);
    VectorIds particle_markers(particle_ids);

    const MPI_Comm comm = dof_handler.get_communicator();

    const unsigned int n_global_levels =
      dof_handler.get_triangulation().n_global_levels();
    const unsigned int max_level = n_global_levels - 1;

    // Estimate cell size
    const unsigned int n_local_levels =
      dof_handler.get_triangulation().n_levels();
    const auto h_cell_local =
      dof_handler.begin_active(n_local_levels - 1)->diameter() / std::sqrt(dim);

    const auto h_cell = Utilities::MPI::min<double>(h_cell_local, comm);

    /* This container stores the groups of cells forming a kind of iso-surface
     * at a given distance from each of the particle cliques. As a storage
     * container, std::deque is chosen since later at each iteration we pick the
     * first item out of it and will add a new one to the end, such that the
     * size of this container stays the same. The size equals to the number of
     * levels in the triangulation. */
    std::deque<std::vector<DoFCellAccessor<dim, dim, false>>> agglomerations(
      n_global_levels);

    // Set initial value of the markers
    particle_markers = invalid_particle_id;

    // Run preparatory modified flooding
    particle_ids.update_ghost_values();
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const auto particle_id = particle_ids[cell->global_active_cell_index()];
        const auto particle_marker =
          particle_markers[cell->global_active_cell_index()];

        if (particle_id == invalid_particle_id ||
            particle_marker != invalid_particle_id)
          continue;

        internal::run_flooding_prep(cell,
                                    particle_ids,
                                    particle_markers,
                                    agglomerations,
                                    invalid_particle_id);
      }
    particle_ids.zero_out_ghost_values();

    auto cell_weight = [&max_level](const auto &cell) {
      return std::pow(2, max_level - cell.level());
    };

    // Set zero distances
    particle_distances = invalid_particle_id;
    for (const auto &agglomeration : agglomerations)
      for (const auto &cell : agglomeration)
        particle_distances[cell.global_active_cell_index()] =
          cell_weight(cell) - 1;

    // We store distances here
    std::map<std::pair<unsigned int, unsigned int>, double>
      assessment_distances;

    /* This lambda sets distance for newly colored cells or estimates distance
     * between the two particle cliques if collision has been detected. */
    auto handle_cells = [&max_level,
                         &h_cell,
                         &assessment_distances,
                         &particle_markers,
                         &particle_distances,
                         &invalid_particle_id,
                         &agglomerations,
                         &cell_weight](const auto &cell, const auto &neighbor) {
      const auto cell_particle_id =
        particle_markers[cell.global_active_cell_index()];

      const auto neighbor_particle_id =
        particle_markers[neighbor.global_active_cell_index()];

      if (neighbor_particle_id == invalid_particle_id)
        {
          /* Add to agglomeration. The agglomerations container works as a
           * priority queue here: the cells are distributed with respect to
           * their level, e.g. the finest cells get added right at the beginning
           * of the queue and so on. */
          agglomerations[max_level - neighbor.level()].push_back(neighbor);

          // Update distance
          particle_distances[neighbor.global_active_cell_index()] =
            particle_distances[cell.global_active_cell_index()] +
            cell_weight(neighbor);

          particle_markers[neighbor.global_active_cell_index()] =
            cell_particle_id;
        }
      else if (neighbor_particle_id != cell_particle_id)
        {
          // It means that we meet the neighbourhood of
          // another particle
          const auto current_distance =
            h_cell * (particle_distances[cell.global_active_cell_index()] +
                      particle_distances[neighbor.global_active_cell_index()]);

          const auto key =
            std::make_pair(std::min(cell_particle_id, neighbor_particle_id),
                           std::max(cell_particle_id, neighbor_particle_id));

          auto it = assessment_distances.find(key);
          if (it == assessment_distances.end())
            assessment_distances.try_emplace(key, current_distance);
          else
            it->second = std::min(it->second, current_distance);
        }
    };

    // Cache all ghost cells
    using CellsCache = std::pair<DoFCellAccessor<dim, dim, false>,
                                 std::vector<DoFCellAccessor<dim, dim, false>>>;

    /* This list contains a list of ghost cells that are adjacent to those
     * locally owned cells which do not belong to any particle clique. One any
     * of such ghost cells has been colored, we will then transfer this
     * information to the adjacent locally owned cells. This mechanism is
     * required to capture grows of cliques across different ranks. */
    std::list<CellsCache> all_ghost_cells;

    // Crucial - otherwise zeros are returned
    particle_markers.update_ghost_values();

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_ghost())
        {
          CellsCache cache(*cell, {});

          for (const auto face : cell->face_indices())
            if (!cell->at_boundary(face))
              {
                const auto neighbor = cell->neighbor(face);

                if (!neighbor->has_children())
                  {
                    if (neighbor->is_locally_owned())
                      {
                        const auto neighbor_particle_id =
                          particle_markers[neighbor
                                             ->global_active_cell_index()];

                        if (neighbor_particle_id == invalid_particle_id)
                          cache.second.push_back(*neighbor);
                      }
                  }
                else
                  {
                    for (const auto &child : neighbor->child_iterators())
                      {
                        /* If a child has its own children, then for sure this
                         * is not a cell that is adjacent to the current one, so
                         * we skip it, that is why is_active() is here for. */
                        if (child->is_active() && child->is_locally_owned() &&
                            particle_markers[child
                                               ->global_active_cell_index()] ==
                              invalid_particle_id)
                          for (const auto child_face : child->face_indices())
                            if (!child->at_boundary(child_face))
                              {
                                const auto neighbor_of_child =
                                  child->neighbor(child_face);

                                if (neighbor_of_child->is_active() &&
                                    !neighbor_of_child->is_artificial() &&
                                    neighbor_of_child->id() == cell->id())
                                  cache.second.push_back(*child);
                              }
                      }
                  }
              }

          // Add only if we have some cell candidates in cache
          if (!cache.second.empty())
            all_ghost_cells.push_back(cache);
        }

    // Some helpful lambdas
    auto check_agglomerations = [&agglomerations, &comm]() {
      const bool has_non_empty_agglomerations =
        std::find_if(agglomerations.cbegin(),
                     agglomerations.cend(),
                     [](const auto &agglomeration) {
                       return !agglomeration.empty();
                     }) != agglomerations.end();

      return (Utilities::MPI::sum<unsigned int>(has_non_empty_agglomerations,
                                                comm) > 0);
    };

    auto check_ghosts_cache = [&all_ghost_cells, &comm]() {
      return (
        Utilities::MPI::sum<unsigned int>(!all_ghost_cells.empty(), comm) > 0);
    };

    /* We iterate over agglomerations moving in layerwise manner from the
     * particles cliques towards the voids of the computational domain until two
     * growing cliques meet or a boundary is encountered. At each iteration,
     * when a new cell is colored, its distance in general is incremented with
     * respect to its neighbor that was colored on one of the previous
     * iterations. The "pace" for a cell depends on its refinement level that
     * defines a weight of a cell contributing to the distance evaluations: the
     * finest cells have weight equals to 1 and larger ones have bigger weights,
     * see below the formula. */
    bool       do_process_agglomerations = check_agglomerations();
    bool       do_update_ghosts          = check_ghosts_cache();
    const bool need_to_zero_out          = do_update_ghosts;
    while (do_process_agglomerations)
      {
        // Pick the nearest agglomeration set - copy it
        const auto agglomeration_at_level = agglomerations.front();
        agglomerations.pop_front();

        // Add empty
        agglomerations.emplace_back(
          std::vector<DoFCellAccessor<dim, dim, false>>());

        if (do_update_ghosts)
          {
            particle_markers.update_ghost_values();
            particle_distances.update_ghost_values();
          }

        /* Run over ghost elements only, we need to transfer infromation across
         * ranks if somewhere a growing clique has reached any neighbor. */
        auto it_cache = all_ghost_cells.begin();
        while (it_cache != all_ghost_cells.end())
          {
            const auto &cache      = *it_cache;
            const auto &ghost_cell = cache.first;

            const auto ghost_particle_id =
              particle_markers[ghost_cell.global_active_cell_index()];

            if (ghost_particle_id != invalid_particle_id)
              {
                for (const auto &local_cell : cache.second)
                  {
                    const auto local_particle_id =
                      particle_markers[local_cell.global_active_cell_index()];

                    // Check whether this cell has already be assigned
                    if (local_particle_id == invalid_particle_id)
                      {
                        // We add new local cells as independet agglomerations
                        agglomerations[max_level - local_cell.level()]
                          .push_back(local_cell);

                        // Set distance for the newly colored cell
                        particle_distances[local_cell
                                             .global_active_cell_index()] =
                          particle_distances[ghost_cell
                                               .global_active_cell_index()] +
                          cell_weight(local_cell);

                        // Mark the cell as visited
                        particle_markers[local_cell
                                           .global_active_cell_index()] =
                          ghost_particle_id;
                      }
                  }

                // Delete this neighbor data as we do not need it anymore
                it_cache = all_ghost_cells.erase(it_cache);
              }
            else
              {
                ++it_cache;
              }
          }

        // Run over the previously picked agglomerations.
        for (const auto &cell : agglomeration_at_level)
          {
            for (const auto face : cell.face_indices())
              if (!cell.at_boundary(face))
                {
                  const auto neighbor = cell.neighbor(face);

                  if (!neighbor->has_children())
                    {
                      if (neighbor->is_locally_owned())
                        handle_cells(cell, *neighbor);
                    }
                  else
                    {
                      for (const auto &child : neighbor->child_iterators())
                        if (child->is_active() && child->is_locally_owned())
                          for (const auto child_face : child->face_indices())
                            if (!child->at_boundary(child_face))
                              {
                                const auto neighbor_of_child =
                                  child->neighbor(child_face);

                                if (neighbor_of_child->is_active() &&
                                    neighbor_of_child->is_locally_owned() &&
                                    neighbor_of_child->id() == cell.id())
                                  {
                                    handle_cells(cell, *child);
                                    break;
                                  }
                              }
                    }
                }
          }

        do_process_agglomerations = check_agglomerations();
        do_update_ghosts          = check_ghosts_cache();
      }

    particle_markers.zero_out_ghost_values();
    if (need_to_zero_out)
      particle_distances.zero_out_ghost_values();

    // Convert map to a vector
    std::vector<double> distances_flatten;
    for (const auto &[from_to, dist] : assessment_distances)
      {
        distances_flatten.push_back(from_to.first);
        distances_flatten.push_back(from_to.second);
        distances_flatten.push_back(dist);
      }

    // Perform global communication, the data is not large
    const auto global_distances =
      Utilities::MPI::all_gather(comm, distances_flatten);

    assessment_distances.clear();
    for (const auto &distances_set : global_distances)
      for (unsigned int i = 0; i < distances_set.size(); i += 3)
        {
          const auto key =
            std::make_pair(static_cast<unsigned int>(distances_set[i]),
                           static_cast<unsigned int>(distances_set[i + 1]));

          auto it = assessment_distances.find(key);
          if (it == assessment_distances.end())
            assessment_distances.try_emplace(key, distances_set[i + 2]);
          else
            it->second = std::min(it->second, distances_set[i + 2]);
        }

    // Output the distance and marker vectors for debug purposes
    if (data_out)
      {
        Vector<double> particle_distances_local(
          dof_handler.get_triangulation().n_active_cells());
        Vector<double> particle_markers_local(
          dof_handler.get_triangulation().n_active_cells());
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              particle_distances_local[cell->active_cell_index()] =
                particle_distances[cell->global_active_cell_index()];
              particle_markers_local[cell->active_cell_index()] =
                particle_markers[cell->global_active_cell_index()];
            }

        data_out->add_data_vector(particle_distances_local,
                                  "particle_distances",
                                  DataOut<dim>::DataVectorType::type_cell_data);
        data_out->add_data_vector(particle_markers_local,
                                  "particle_markers",
                                  DataOut<dim>::DataVectorType::type_cell_data);
      }

    return assessment_distances;
  }

  /* Detect particle ids across multiple order parameters which interact with
   * each other via a direct contact. The two particles interact with each other
   * if their ids sit on the same or the first immediate neighbor cell. */
  template <int dim, typename BlockVectorIds>
  std::map<std::pair<unsigned int, unsigned int>,
           std::set<std::pair<unsigned int, unsigned int>>>
  get_direct_neighbors(const DoFHandler<dim> &dof_handler,
                       const BlockVectorIds  &particle_ids,
                       const double           invalid_particle_id = -1.0)
  {
    /* Each pair here is (order_parameter_id, global_particle_id), where
     * particle ids are evaluated per order parameter, for this reason we need
     * to store also the order parameter index. */
    std::map<std::pair<unsigned int, unsigned int>,
             std::set<std::pair<unsigned int, unsigned int>>>
      neighbors;

    auto add_neighbors =
      [&neighbors](const std::pair<unsigned int, unsigned int> &pi,
                   const std::pair<unsigned int, unsigned int> &pj) {
        neighbors[pi].insert(pj);
        neighbors[pj].insert(pi);
      };

    // Compute particles moments of inertia
    for (const auto &cell :
         dof_handler.get_triangulation().active_cell_iterators())
      if (cell->is_locally_owned())
        for (unsigned int i = 0; i < particle_ids.n_blocks(); ++i)
          {
            const auto pid_i =
              particle_ids.block(i)[cell->global_active_cell_index()];

            const auto pi = std::make_pair(i, pid_i);

            if (pid_i != invalid_particle_id)
              for (unsigned int j = i + 1; j < particle_ids.n_blocks(); ++j)
                {
                  const auto pid_j =
                    particle_ids.block(j)[cell->global_active_cell_index()];

                  if (pid_j != invalid_particle_id)
                    {
                      const auto pj = std::make_pair(j, pid_j);

                      add_neighbors(pi, pj);
                    }

                  for (const auto face : cell->face_indices())
                    if (!cell->at_boundary(face))
                      {
                        const auto neighbor = cell->neighbor(face);

                        if (!neighbor->has_children())
                          {
                            const auto neighbor_pid_j = particle_ids.block(
                              j)[neighbor->global_active_cell_index()];

                            if (neighbor_pid_j != invalid_particle_id)
                              {
                                const auto pj =
                                  std::make_pair(j, neighbor_pid_j);

                                add_neighbors(pi, pj);
                              }
                          }
                        else
                          {
                            for (const auto &child :
                                 neighbor->child_iterators())
                              if (child->is_active() && !child->is_artificial())
                                {
                                  const auto child_pid_j = particle_ids.block(
                                    j)[child->global_active_cell_index()];

                                  if (child_pid_j != invalid_particle_id)
                                    {
                                      const auto pj =
                                        std::make_pair(j, child_pid_j);

                                      add_neighbors(pi, pj);
                                    }
                                }
                          }
                      }
                }
          }

    // Gather data globally
    std::vector<unsigned int> neighbors_flatten;

    for (const auto &[primary, secondaries] : neighbors)
      {
        neighbors_flatten.push_back(primary.first);
        neighbors_flatten.push_back(primary.second);
        neighbors_flatten.push_back(secondaries.size());
        for (const auto &secondary : secondaries)
          {
            neighbors_flatten.push_back(secondary.first);
            neighbors_flatten.push_back(secondary.second);
          }
      }

    const auto neighbors_global =
      Utilities::MPI::all_gather(dof_handler.get_communicator(),
                                 neighbors_flatten);

    // Gather results from the communication
    neighbors.clear();
    for (const auto &neighbors_current : neighbors_global)
      for (unsigned int i = 0; i < neighbors_current.size();)
        {
          const auto [it, status] = neighbors.emplace(
            std::make_pair(neighbors_current[i], neighbors_current[i + 1]),
            std::set<std::pair<unsigned int, unsigned int>>());

          const unsigned int n_neighbors = neighbors_current[i + 2];
          for (unsigned int j = 0; j < n_neighbors; ++j)
            it->second.insert(
              std::make_pair(neighbors_current[i + 3 + 2 * j],
                             neighbors_current[i + 3 + 2 * j + 1]));

          i += 3 + 2 * n_neighbors;
        }

    return neighbors;
  }
} // namespace GrainTracker