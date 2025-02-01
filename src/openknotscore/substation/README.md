# Substation
Substation is a a compute/data pipeline framework specifically designed for this project.
It is designed to be useful enough to be applicable outside this project - it may eventually move
into its own package

## Usage
In general, you will start by instantiating a `Runner` (options are found in `substation/runners`).
A runner encapsulates a way of taking some work to be done and sending it to/distributing it across
some compute resources (eg the local machine, a slurm cluster, etc).

Once you have a runner instance, you can create a list of `Task`s (`substation.domain.Task`) to pass
to it, which define some work to be done (ie, a function and its arguments) as well as the resources
required to do so

## Internals
There are four high-level components of substation:
* Runners, described above
* The scheduler (`substation.scheduler.scheduler`). This allows you to, take a set of tasks with diverse
  resource requirements (CPUs, memory, runtime, etc) and create a relatively efficient "schedule"
  for them to run on your compute - eg, if you have a 12-core slurm allocation, you could use the first
  6 cores to run a 10 minute 6-core job, while using the other 6 to run a 2 minute 6-core job then
  6 concurrent 8 minute 1-core jobs.
* The task database (`substation.taskdb`). This is used by runners where tasks will be spread out to many
  remote machines/processes which each need to look up information about the tasks they're supposed
  to run.
* `substation.deferred` - this is a special module where tasks can register that there is some work that
  they would like to be done, but not after every task. For example, when saving results to a database,
  instead of inserting after every task, you can batch inserts together (every so many tasks, after
  a minimum of so much time has elapsed since the last time it was run, etc). The runner is responsible
  for making sure these deferred tasks are run when required (including before the program exits)

Note that tasks are scheduled in the following hierarchy:
* ComputeConfiguration: Defines a configuration of resources, eg the size of a cloud instance or the
  configuration of a slurm job
* ComputeAllocation: A concrete "instance" of a compute configuration - eg, for a slurm job array,
  that array/sbatch command is one compute configuration, and once compute allocation per array job
* TaskQueue: A a "slice" of a ComputeAllocation's resources and the tasks to be run on it.
  Queues are run in parallel, with each queue running its tasks serially
  Eg, a 12-core allocation could have a 6 core queue plus two 3-core queues, each with five tasks.
  At any given time, one task from each queue will be running (unless the queue is exhausted).
  After all jobs in a queue are complete, it can also be "split", eg after the tasks in the 6-core
  queue finish, in its place there could be 3 2-core queues.
* Task: As described earlier
