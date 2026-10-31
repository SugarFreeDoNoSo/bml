"""Run an owned named container and confirm cleanup before returning or raising."""
import subprocess
import uuid


class ContainerCleanupError(RuntimeError):
    """Cleanup was not confirmed; callers must stop rather than start more work."""

    def __init__(self, message, execution_returncode=None):
        super().__init__(message)
        self.execution_returncode = execution_returncode


def run_container(cmd, **kwargs):
    run_index = cmd.index('run')
    prefix = list(cmd[:run_index])
    if '--name' in cmd[run_index+1:]:
        raise ValueError('run_container assigns its own unique container name')
    name = 'bml-owned-' + uuid.uuid4().hex
    named = prefix + ['run', '--name', name] + list(cmd[run_index+1:])
    result = None
    try:
        result = subprocess.run(named, **kwargs)
        return result
    finally:
        # TimeoutExpired only kills the client; explicitly remove this container.
        # --rm may already have removed it after normal completion.
        try:
            cleanup = subprocess.run(prefix + ['rm', '-f', name], capture_output=True,
                                     text=True, env=kwargs.get('env'), timeout=20)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ContainerCleanupError('container cleanup unconfirmed: '+name,
                                        getattr(result, 'returncode', None)) from error
        absent = cleanup.returncode == 1 and cleanup.stderr.strip().endswith('No such container: '+name)
        if cleanup.returncode != 0 and not absent:
            raise ContainerCleanupError(
                f'container cleanup failed ({cleanup.returncode}): {name}: {cleanup.stderr[:300]}',
                getattr(result, 'returncode', None))
