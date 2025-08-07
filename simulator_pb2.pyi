from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Action(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_UNSPECIFIED: _ClassVar[Action]
    ACTION_WAIT: _ClassVar[Action]
    ACTION_SEND_PRIMARY: _ClassVar[Action]
    ACTION_SEND_BACKUP: _ClassVar[Action]

class Priority(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PRIORITY_UNSPECIFIED: _ClassVar[Priority]
    PRIORITY_LOW: _ClassVar[Priority]
    PRIORITY_MEDIUM: _ClassVar[Priority]
    PRIORITY_HIGH: _ClassVar[Priority]
    PRIORITY_CRITICAL: _ClassVar[Priority]
ACTION_UNSPECIFIED: Action
ACTION_WAIT: Action
ACTION_SEND_PRIMARY: Action
ACTION_SEND_BACKUP: Action
PRIORITY_UNSPECIFIED: Priority
PRIORITY_LOW: Priority
PRIORITY_MEDIUM: Priority
PRIORITY_HIGH: Priority
PRIORITY_CRITICAL: Priority

class AgentObservation(_message.Message):
    __slots__ = ("has_message", "top_message_priority", "primary_channel_busy", "backup_channel_busy", "pending_acks_count", "outbound_queue_length")
    HAS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TOP_MESSAGE_PRIORITY_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_CHANNEL_BUSY_FIELD_NUMBER: _ClassVar[int]
    BACKUP_CHANNEL_BUSY_FIELD_NUMBER: _ClassVar[int]
    PENDING_ACKS_COUNT_FIELD_NUMBER: _ClassVar[int]
    OUTBOUND_QUEUE_LENGTH_FIELD_NUMBER: _ClassVar[int]
    has_message: bool
    top_message_priority: Priority
    primary_channel_busy: bool
    backup_channel_busy: bool
    pending_acks_count: int
    outbound_queue_length: int
    def __init__(self, has_message: bool = ..., top_message_priority: _Optional[_Union[Priority, str]] = ..., primary_channel_busy: bool = ..., backup_channel_busy: bool = ..., pending_acks_count: _Optional[int] = ..., outbound_queue_length: _Optional[int] = ...) -> None: ...

class AgentState(_message.Message):
    __slots__ = ("observation", "reward", "done")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    observation: AgentObservation
    reward: float
    done: bool
    def __init__(self, observation: _Optional[_Union[AgentObservation, _Mapping]] = ..., reward: _Optional[float] = ..., done: bool = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("actions",)
    class ActionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Action
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Action, str]] = ...) -> None: ...
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    actions: _containers.ScalarMap[str, Action]
    def __init__(self, actions: _Optional[_Mapping[str, Action]] = ...) -> None: ...

class StepResponse(_message.Message):
    __slots__ = ("states",)
    class StatesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: AgentState
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[AgentState, _Mapping]] = ...) -> None: ...
    STATES_FIELD_NUMBER: _ClassVar[int]
    states: _containers.MessageMap[str, AgentState]
    def __init__(self, states: _Optional[_Mapping[str, AgentState]] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("states",)
    class StatesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: AgentState
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[AgentState, _Mapping]] = ...) -> None: ...
    STATES_FIELD_NUMBER: _ClassVar[int]
    states: _containers.MessageMap[str, AgentState]
    def __init__(self, states: _Optional[_Mapping[str, AgentState]] = ...) -> None: ...
