// ---------------------------------------------------------------------
//
// Copyright (C) 2025 - 2026 by the hpsint authors
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

#include <pf-applications/base/timer.h>

#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>

using namespace dealii;

void
MemoryTracker::enable(const bool flag_)
{
  flag() = flag_;
}

bool
MemoryTracker::enabled()
{
  return flag();
}

double
MemoryTracker::current_rss_gb()
{
  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);

  return stats.VmRSS / 1e6;
}

void
MemoryTracker::record(const std::string &section, const double delta_gb)
{
  auto &entry = data()[section];
  entry.first += delta_gb;
  entry.second += 1;
}

void
MemoryTracker::print(std::ostream &out, const unsigned int n_entries)
{
  // Sections may differ between ranks, so gather them on rank 0
  const auto all_data = Utilities::MPI::gather(MPI_COMM_WORLD, data(), 0);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
    return;

  struct Entry
  {
    double       sum     = 0.;
    double       max     = 0.;
    unsigned int n_calls = 0;
  };

  std::map<std::string, Entry> entries;
  for (const auto &rank_data : all_data)
    for (const auto &[section, delta_and_calls] : rank_data)
      {
        auto &entry = entries[section];
        entry.sum += delta_and_calls.first;
        entry.max     = std::max(entry.max, delta_and_calls.first);
        entry.n_calls = std::max(entry.n_calls, delta_and_calls.second);
      }

  std::vector<std::pair<std::string, Entry>> sorted(entries.begin(),
                                                    entries.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
    return a.second.sum > b.second.sum;
  });

  out << "RSS growth per section [GB] (sum over ranks / max rank / calls):"
      << std::endl;
  for (unsigned int i = 0; i < std::min<std::size_t>(n_entries, sorted.size());
       ++i)
    out << "  " << std::setw(12) << sorted[i].second.sum << " / "
        << std::setw(10) << sorted[i].second.max << " / " << std::setw(8)
        << sorted[i].second.n_calls << "  " << sorted[i].first << std::endl;
}

std::map<std::string, std::pair<double, unsigned int>> &
MemoryTracker::data()
{
  static std::map<std::string, std::pair<double, unsigned int>> data;
  return data;
}

bool &
MemoryTracker::flag()
{
  static bool flag = false;
  return flag;
}

void
monitor(const std::string label)
{
  // return;

  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  pcout << "MONITOR " << label << ": ";

  if (label != "break")
    {
      const auto print = [&pcout](const double value) {
        const auto min_max_avg =
          Utilities::MPI::min_max_avg(value / 1e6, MPI_COMM_WORLD);

        pcout << min_max_avg.min << " " << min_max_avg.max << " "
              << min_max_avg.avg << " " << min_max_avg.sum << " ";
      };

      print(stats.VmPeak);
      print(stats.VmSize);
      print(stats.VmHWM);
      print(stats.VmRSS);
    }

  pcout << std::endl;
}


TimerPredicate::Type
TimerPredicate::string_to_enum(const std::string label)
{
  if (label == "never")
    return never;
  if (label == "n_calls")
    return n_calls;
  if (label == "real_time")
    return real_time;
  if (label == "simulation_time")
    return simulation_time;

  AssertThrow(false, ExcNotImplemented());

  return never;
}

TimerPredicate::TimerPredicate(const std::string label,
                               const double      start,
                               const double      interval)
{
  reinit(string_to_enum(label), start, interval);
}

TimerPredicate::TimerPredicate(const Type   type,
                               const double start,
                               const double interval)
{
  reinit(type, start, interval);
}

void
TimerPredicate::reinit(const Type   type,
                       const double start,
                       const double interval)
{
  this->counter              = 0;
  this->last_simulation_time = 0;

  this->type = type;

  if (type == Type::never)
    {
      // nothing to do
    }
  else if (type == Type::n_calls)
    {
      this->counter  = static_cast<unsigned int>(start);
      this->interval = interval;
    }
  else if (type == Type::simulation_time)
    {
      this->last_simulation_time = start;
      this->interval             = interval;
    }
  else if (type == Type::real_time)
    {
      this->last_real_time = std::chrono::system_clock::now();
      this->interval       = interval;
    }
  else
    {
      Assert(false, ExcNotImplemented());
    }
}

bool
TimerPredicate::now(const double time)
{
  ++counter;

  if (type == Type::never)
    {
      return false;
    }
  else if (type == Type::n_calls)
    {
      if (interval <= 0.0)
        return false;

      return ((counter - 1) % static_cast<unsigned int>(interval)) == 0;
    }
  else if (type == Type::simulation_time)
    {
      if (interval <= 0.0)
        return false;

      if ((time - last_simulation_time - interval) > -1e-10)
        last_simulation_time = time;
      return last_simulation_time == time;
    }
  else if (type == Type::real_time)
    {
      if (interval <= 0.0)
        return false;

      const bool do_output =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now() - last_real_time)
            .count() /
          1e9 >
        interval;

      if (Utilities::MPI::sum<unsigned int>(do_output, MPI_COMM_WORLD) == 0)
        return false;

      last_real_time = std::chrono::system_clock::now();
      return true;
    }
  else
    {
      Assert(false, ExcNotImplemented());
      return false;
    }
}

void
TimerCollection::attach(const MyTimerOutput &timer)
{
  static int counter = 0;

  get_instance().timers.emplace(counter++, &timer);
}

void
TimerCollection::detach(const MyTimerOutput &timer)
{
  auto &set = get_instance().timers;

  const auto ptr =
    std::find_if(set.begin(), set.end(), [&timer](const auto &p) {
      return p.second == &timer;
    });

  Assert(ptr != set.end(),
         ExcMessage("Timer could not be found! Have you attached it?"));

  set.erase(ptr);
}

TimerCollection &
TimerCollection::get_instance()
{
  static TimerCollection instance;

  return instance;
}

void
TimerCollection::configure(const double interval)
{
  get_instance().predicate.reinit(TimerPredicate::real_time,
                                  0.0 /*dummy value*/,
                                  interval);
  get_instance().enabled = interval > 0.0;
}

bool
TimerCollection::timing_enabled()
{
  return get_instance().enabled;
}

TimerCollection::TimerCollection()
{}

void
TimerCollection::print_all_wall_time_statistics(const bool force_output)
{
  if (force_output || get_instance().predicate.now())
    for (const auto &timer : get_instance().timers)
      timer.second->print_wall_time_statistics();
}


MyTimerOutput::MyTimerOutput(std::ostream &out, const bool enabled)
  : pcout(out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
  , enabled(enabled)
{
  if (enabled)
    TimerCollection::attach(*this);
}

MyTimerOutput::MyTimerOutput(const bool enabled)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
  , enabled(enabled)
{
  if (enabled)
    TimerCollection::attach(*this);
}

MyTimerOutput::~MyTimerOutput()
{
  if (enabled)
    TimerCollection::detach(*this);
}

TimerOutput &
MyTimerOutput::operator()()
{
  return timer;
}

const TimerOutput &
MyTimerOutput::operator()() const
{
  return timer;
}

void
MyTimerOutput::enter_subsection(const std::string label)
{
  if (TimerCollection::timing_enabled())
    timer.enter_subsection(label);
}

void
MyTimerOutput::leave_subsection(const std::string label)
{
  if (TimerCollection::timing_enabled())
    timer.leave_subsection(label);
}

void
MyTimerOutput::print_wall_time_statistics() const
{
  Assert(enabled, ExcInternalError());

  // The statistics are collective, so all ranks have to agree on printing
  const unsigned int has_data =
    timer.get_summary_data(TimerOutput::OutputData::total_wall_time).size() > 0;
  if (Utilities::MPI::max(has_data, MPI_COMM_WORLD) > 0)
    timer.print_wall_time_statistics(MPI_COMM_WORLD);
}

bool
MyTimerOutput::is_enabled() const
{
  return enabled;
}



MyScope::MyScope(TimerOutput       &timer_,
                 const std::string &section_name,
                 const bool         do_timing)
  : section_name(section_name)
{
  if (do_timing && TimerCollection::timing_enabled())
    scope = std::make_unique<TimerOutput::Scope>(timer_, section_name);
  enter();
}

MyScope::MyScope(MyTimerOutput     &timer_,
                 const std::string &section_name,
                 const bool         do_timing)
  : MyScope(timer_(), section_name, do_timing)
{}

MyScope::MyScope(const std::string &section_name, MyTimerOutput *ptr_timer)
  : section_name(section_name)
{
  if (ptr_timer && TimerCollection::timing_enabled())
    scope = std::make_unique<TimerOutput::Scope>((*ptr_timer)(), section_name);
  enter();
}

MyScope::~MyScope()
{
  leave();
}

void
MyScope::enter()
{
  if (MemoryTracker::enabled())
    rss_enter = MemoryTracker::current_rss_gb();

#ifdef WITH_TIMING_OUTPUT
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << ">>> MyScope: entering <" << section_name << ">" << std::endl;
  monitor(section_name + "_in_");
#endif
}

void
MyScope::leave()
{
  if (MemoryTracker::enabled())
    MemoryTracker::record(section_name,
                          MemoryTracker::current_rss_gb() - rss_enter);

#ifdef WITH_TIMING_OUTPUT
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << ">>> MyScope: leaving <" << section_name << ">" << std::endl;
  monitor(section_name + "_out");
#endif
}
