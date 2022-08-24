"""Store the state of currently executing graph instances in redis."""
from . import State, InstanceState, InstanceDetail
from ..graph import InstanceId, GraphInstance
from .. import util

from typing import Optional, Mapping, Any
from collections import namedtuple
from enum import Enum
import redis


class RedisEventType(Enum):
    INSTANCE_STEP = 'INSTANCE_STEP'
    STATE_CHANGE = 'STATE_CHANGE'


class RedisEvent(namedtuple('RedisEvent',
                            ('type', 'instance',
                             'new_state',
                             'completed_ops',),
                            defaults=(None, None,))):

    @classmethod
    def __readjson__(self, d: Any, dec: util.JSONDecoder):
        d['type'] = dec.as_(d['type'], RedisEventType)
        return self(**d)


class RedisState(State):
    """Keep track of graph instances on a Redis process."""

    __slots__ = ('client',)

    client: redis.Redis

    def __init__(self, **kwargs):
        """Initialize the state from a set of redis connection parameters."""
        if 'client' in kwargs:
            self.client = kwargs['client']
        else:
            self.client = redis.Redis(**kwargs)

    @classmethod
    def _instance_key(self, instance: InstanceId):
        return f"i-{instance}"

    @classmethod
    def _state_key(self, instance: InstanceId):
        ikey = self._instance_key(instance)
        return f"{ikey}-state"

    @classmethod
    def _meta_key(self, instance: InstanceId):
        ikey = self._instance_key(instance)
        return f"{ikey}-meta"

    @classmethod
    def _complete_ports_key(self, instance: InstanceId):
        ikey = self._instance_key(instance)
        return f"{ikey}-complete"

    @classmethod
    def _op_key(self, instance: InstanceId, opid: int, portname: str):
        ikey = self._instance_key(instance)
        return f"{ikey}-{opid}-{portname}"

    @classmethod
    def _pub_key(self, instance: InstanceId):
        ikey = self._instance_key(instance)
        return f"{ikey}-notify"

    def step_instance_state(self, instance: InstanceId, op: int, complete: bool,
                            portname: Optional[str] = None):
        """Step instance state atomically."""
        if not complete:
            return # If this is not a state update where we're marking a node complete.. ignore

        pipe = self.client.pipeline()

        completed = []
        if portname is None:
            meta = util.loads_json(self.client.get(self._meta_key(instance)))
            if 'instance' not in meta:
                raise RuntimeError("Can't step state when there's no instance")

            base_instance: GraphInstance = util.loads_json(meta['instance'], GraphInstance, repo=self._context.repo)
            for o in base_instance[op].get_signature().outputs:
                pipe.sadd(self._complete_ports_key(instance), f"{op},{o}")
                completed.append((op, o))
        else:
            pipe.sadd(self._complete_ports_key(instance), f"{op},{portname}")
            completed.append((op, portname))

        event = RedisEvent(type=RedisEventType.INSTANCE_STEP,
                           instance=instance,
                           completed_ops=completed)
        pipe.publish(self._pub_key(instance), util.dumps_json(event))

        pipe.execute()

    def get_instance_state(self, instance: InstanceId) -> Optional[InstanceDetail]:
        """Retrieve the instance state."""
        stkey = self._state_key(instance)
        metakey = self._meta_key(instance)
        portkey = self._complete_ports_key(instance)

        pipe = self.client.pipeline()
        pipe.get(stkey)
        pipe.get(metakey)
        pipe.smembers(portkey)
        state, meta, complete = pipe.execute()

        meta = util.loads_json(meta)

        if 'instance' in meta:
            base_instance = util.loads_json(meta['instance'], GraphInstance, repo=self._context.repo)

            # Now list all completed ports
            for p in complete:
                opid, portname = p.decode('utf-8').split(",", 1)
                opid = int(opid)
                base_instance.mark_complete(opid, portname)

            meta['instance'] = util.dumps_json(base_instance)

        return InstanceDetail(state=InstanceState(state.decode('utf-8')),
                              metadata=meta)

    def set_instance_state(self, instance: InstanceId, state: InstanceState,
                           metadata: Mapping[str, Any]):
        """Set the instance state."""
        pipe = self.client.pipeline()

        pipe.set(self._meta_key(instance), util.dumps_json(metadata))
        pipe.set(self._state_key(instance), state.value)

        event = RedisEvent(type=RedisEventType.STATE_CHANGE,
                           instance=instance,
                           new_state=state)
        pipe.publish(self._pub_key(instance),
                     util.dumps_json(event))

        pipe.execute()

    def save_op_result(self, instance: InstanceId, opid: int, portname: str,
                       result: Any, order: Optional[int] = None):
        """Save the given result pointer as a result of the given op in the given instance."""
        opkey = self._op_key(instance, opid, portname)
        if order is not None:
            pipe = self.client.pipeline()
            pipe.hset(opkey, str(order), util.dumps_json(result))
        else:
            self.client.set(opkey, util.dumps_json(result))

    def load_op_result(self, instance: InstanceId, opid: int, portname: str) -> Any:
        """Load the given op result pointer."""
        opkey = self._op_key(instance, opid, portname)
        oty = self.client.type(opkey)
        if oty == 'hash':
            opval = self.client.hgetall(opkey)
            res: list[Any] = []

            for order in opval:
                try:
                    order = int(order)
                except ValueError:
                    continue

                if order < 0:
                    continue

                if order >= len(res):
                    new_elems = order - len(res) + 1
                    res.extend(None for _ in range(new_elems))

                res[order] = util.loads_json(opval[order], repo=self._context.repo)

            if any(o is None for o in res):
                raise ValueError("Incomplete dynamic output requested")

            return res
        else:
            o = self.client.get(opkey)
            return util.loads_json(o, repo=self._context.repo)

    def wait_for_update(self, instance: InstanceId):
        """Wait for an update to the given instance."""
        ps = self.client.pubsub(ignore_subscribe_messages=True)
        ps.subscribe(self._pub_key(instance))
        for msg in ps.listen():
            return
