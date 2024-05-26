# generated from rosidl_generator_py/resource/_idl.py.em
# with input from custom_interfaces:srv/TurningControl.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_TurningControl_Request(type):
    """Metaclass of message 'TurningControl_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('custom_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'custom_interfaces.srv.TurningControl_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__turning_control__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__turning_control__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__turning_control__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__turning_control__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__turning_control__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class TurningControl_Request(metaclass=Metaclass_TurningControl_Request):
    """Message class 'TurningControl_Request'."""

    __slots__ = [
        '_time_duration',
        '_angular_vel_z',
        '_linear_vel_x',
    ]

    _fields_and_field_types = {
        'time_duration': 'uint32',
        'angular_vel_z': 'double',
        'linear_vel_x': 'double',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.time_duration = kwargs.get('time_duration', int())
        self.angular_vel_z = kwargs.get('angular_vel_z', float())
        self.linear_vel_x = kwargs.get('linear_vel_x', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.time_duration != other.time_duration:
            return False
        if self.angular_vel_z != other.angular_vel_z:
            return False
        if self.linear_vel_x != other.linear_vel_x:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def time_duration(self):
        """Message field 'time_duration'."""
        return self._time_duration

    @time_duration.setter
    def time_duration(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'time_duration' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'time_duration' field must be an unsigned integer in [0, 4294967295]"
        self._time_duration = value

    @property
    def angular_vel_z(self):
        """Message field 'angular_vel_z'."""
        return self._angular_vel_z

    @angular_vel_z.setter
    def angular_vel_z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'angular_vel_z' field must be of type 'float'"
        self._angular_vel_z = value

    @property
    def linear_vel_x(self):
        """Message field 'linear_vel_x'."""
        return self._linear_vel_x

    @linear_vel_x.setter
    def linear_vel_x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'linear_vel_x' field must be of type 'float'"
        self._linear_vel_x = value


# Import statements for member types

# already imported above
# import rosidl_parser.definition


class Metaclass_TurningControl_Response(type):
    """Metaclass of message 'TurningControl_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('custom_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'custom_interfaces.srv.TurningControl_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__turning_control__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__turning_control__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__turning_control__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__turning_control__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__turning_control__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class TurningControl_Response(metaclass=Metaclass_TurningControl_Response):
    """Message class 'TurningControl_Response'."""

    __slots__ = [
        '_success',
    ]

    _fields_and_field_types = {
        'success': 'boolean',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('boolean'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.success = kwargs.get('success', bool())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.success != other.success:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def success(self):
        """Message field 'success'."""
        return self._success

    @success.setter
    def success(self, value):
        if __debug__:
            assert \
                isinstance(value, bool), \
                "The 'success' field must be of type 'bool'"
        self._success = value


class Metaclass_TurningControl(type):
    """Metaclass of service 'TurningControl'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('custom_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'custom_interfaces.srv.TurningControl')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__turning_control

            from custom_interfaces.srv import _turning_control
            if _turning_control.Metaclass_TurningControl_Request._TYPE_SUPPORT is None:
                _turning_control.Metaclass_TurningControl_Request.__import_type_support__()
            if _turning_control.Metaclass_TurningControl_Response._TYPE_SUPPORT is None:
                _turning_control.Metaclass_TurningControl_Response.__import_type_support__()


class TurningControl(metaclass=Metaclass_TurningControl):
    from custom_interfaces.srv._turning_control import TurningControl_Request as Request
    from custom_interfaces.srv._turning_control import TurningControl_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
