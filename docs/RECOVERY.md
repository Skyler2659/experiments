# Recovery contract

The model wrapper resumes the same CLI session after classified transport, capacity,
quota or authentication failures. Exponential backoff begins at 15 seconds. In-flight
tools are drained before continuation. The live Python kernel and simulator remain
owned by the episode process; robot actions are not blindly replayed.

Reconstruction checkpoints validate node assignments, source and child graphs before
reuse. Concurrent provider waiting intervals are unioned before subtracting them from
the active reconstruction budget. Completed child assets remain in place.

`worldrua run CONFIG --detach` starts a service that can restart the queue coordinator.
The coordinator adopts live per-episode owners using PID plus process start time. A
per-episode file lock and persistent owner record prevent a duplicate launch. After
SSH or coordinator loss, use `worldrua resume OUTPUT --detach`. Telemetry delay alone
is not evidence that the robot stopped.

States are distinct:

- `QUEUED`: not started.
- `RUNNING`: live episode, with startup/reconstruction/policy stage.
- `FINISHED`: normal evaluation completion; `success` can be true or false.
- `INTERRUPTED`: infrastructure/owner termination without a valid normal completion.
- `CANCELLED`: explicit cancellation, not a benchmark failure.

Provider journals independently show `RUNNING`, `WAITING_PROVIDER`, `WAITING_TOOL`,
`NEEDS_ATTENTION`, `COMPLETED` or `FAILED`. Keep their timestamps when reporting status.

`pause-queue` affects queued launches only. To stop restarting the coordinator after
its next exit, create `OUTPUT/STOP_SERVICE`. Neither operation kills active robots.
An intentional episode termination needs separate process management and must be
recorded as an interruption/cancellation. New attempts require new output directories.

## Evidence and limits

The original deployed wrapper passed synthetic recovery tests and a real policy-stage
network disturbance: a second CLI attempt resumed the same session, kept the simulator
and 111 completed reconstruction assets, and continued after the first key strike
without replaying it. A reconstruction-stage disturbance also retained progress, but
did not force the outer retry branch. The policy injection was between actions, not
an in-flight action with a lost response. The latter is covered by tool-drain tests.

The packaging scheduler and CLI have local regression tests. They have not yet been
qualified by a fresh evaluated GPU batch from this new repository. Provider quota
recovery has synthetic coverage, not a full real exhaustion-and-reset trial.

OOM, SIGKILL or host reboot killing the simulator cannot be recovered losslessly
without a physics checkpoint. Invalid credentials need a human to renew the selected
login. Unclassified code errors are recorded rather than retried forever. Model task
failures and evaluator failures are not repaired by transport recovery.
