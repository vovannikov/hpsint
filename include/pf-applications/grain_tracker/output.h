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

#pragma once

#include "grain.h"
#include "tracking.h"

namespace GrainTracker
{
  using namespace dealii;

  // Print a single grain
  template <int dim, typename Stream>
  void
  print_grain(const Grain<dim> &grain, Stream &out)
  {
    out << "op_index_current = " << grain.get_order_parameter_id()
        << " | op_index_old = " << grain.get_old_order_parameter_id()
        << " | segments = " << grain.get_segments().size()
        << " | grain_index = " << grain.get_grain_id() << std::endl;

    for (const auto &segment : grain.get_segments())
      {
        out << "    segment: ";
        segment.print(out);
        out << std::endl;
      }
  }

  // Print grains
  template <int dim, typename Stream>
  void
  print_grains(const std::map<unsigned int, Grain<dim>> &grains, Stream &out)
  {
    out << "Number of order parameters: "
        << extract_active_order_parameter_ids(grains).size() << std::endl;
    out << "Number of grains: " << grains.size() << std::endl;
    for (const auto &[gid, gr] : grains)
      {
        (void)gid;
        print_grain(gr, out);
      }
  }

  // Print current grains ordered according to segments location
  template <int dim, typename Stream>
  void
  print_grains_invariant(const std::map<unsigned int, Grain<dim>> &grains,
                         Stream &                                  out)
  {
    std::vector<unsigned int>                         ordered_grains;
    std::map<unsigned int, std::vector<unsigned int>> ordered_segments;

    for (const auto &pair_gid_grain : grains)
      {
        const auto &grain_id = pair_gid_grain.first;
        const auto &grain    = pair_gid_grain.second;

        ordered_grains.push_back(grain_id);

        ordered_segments.emplace(grain_id, std::vector<unsigned int>());
        for (unsigned int i = 0; i < grain.get_segments().size(); i++)
          {
            ordered_segments.at(grain_id).push_back(i);
          }

        std::sort(ordered_segments.at(grain_id).begin(),
                  ordered_segments.at(grain_id).end(),
                  [&grain](const auto &segment_a_id, const auto &segment_b_id) {
                    const auto &segment_a = grain.get_segments()[segment_a_id];
                    const auto &segment_b = grain.get_segments()[segment_b_id];

                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        if (segment_a.get_center()[d] !=
                            segment_b.get_center()[d])
                          {
                            return segment_a.get_center()[d] <
                                   segment_b.get_center()[d];
                          }
                      }
                    return false;
                  });
      }

    std::sort(
      ordered_grains.begin(),
      ordered_grains.end(),
      [&grains, &ordered_segments](const auto &grain_a_id,
                                   const auto &grain_b_id) {
        const auto &grain_a = grains.at(grain_a_id);
        const auto &grain_b = grains.at(grain_b_id);

        const auto &min_segment_a =
          grain_a
            .get_segments()[ordered_segments.at(grain_a.get_grain_id())[0]];
        const auto &min_segment_b =
          grain_b
            .get_segments()[ordered_segments.at(grain_b.get_grain_id())[0]];

        for (unsigned int d = 0; d < dim; ++d)
          {
            if (min_segment_a.get_center()[d] != min_segment_b.get_center()[d])
              {
                return min_segment_a.get_center()[d] <
                       min_segment_b.get_center()[d];
              }
          }
        return false;
      });

    // Printing itself
    out << "Number of order parameters: "
        << extract_active_order_parameter_ids(grains).size() << std::endl;
    out << "Number of grains: " << grains.size() << std::endl;

    for (const auto &grain_id : ordered_grains)
      {
        const auto &grain = grains.at(grain_id);

        out << "op_index_current = " << grain.get_order_parameter_id()
            << " | op_index_old = " << grain.get_old_order_parameter_id()
            << " | segments = " << grain.get_segments().size() << std::endl;

        for (const auto &segment_id : ordered_segments.at(grain_id))
          {
            const auto &segment =
              grains.at(grain_id).get_segments()[segment_id];

            out << "    segment: ";
            segment.print(out);
            out << std::endl;
          }
      }
  }
} // namespace GrainTracker