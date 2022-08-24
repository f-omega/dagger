"""Base class for loggers that log to a SQL database."""

from . import Logger, LogEntry
from ..util import dumps_json

from abc import ABC, abstractmethod
import uuid
import json
import contextlib


class _DbApiValues(object):
    """Represents python values that we want to pass to a generic db api.

    Supports all styles
    """

    __slots__ = ("_data", "_style", "_pos", "args", "kwargs")

    def __init__(self, api, *args, **kwargs):
        """Initialize the values with a PEP249 DB API and dictionaries of arguments."""
        for a in args:
            kwargs.update(**a)

        self._data = kwargs
        self.args = []
        self.kwargs = {}
        self._pos = {}

        self._style = api.paramstyle
        if self._style in ('named', 'pyformat'):
            self.kwargs = self._data

    @property
    def params(self):
        """Get the params value to pass to execute()."""
        if self._style in ('named', 'pyformat'):
            return self.kwargs
        else:
            return self.args

    def __getattr__(self, nm):
        """Get a field as an appropriate parameter."""
        if nm not in self._data:
            raise AttributeError(f"No field {nm} provided")

        if self._style == "named":
            return f":{nm}"
        elif self._style == "qmark":
            self.args.append(self._data[nm])
            return "?"
        elif self._style == "numeric":
            if nm in self._pos:
                pos = self._pos[nm]
            else:
                pos = len(self.args)
                self.args.append(self._data[nm])
            return f":{pos}"
        elif self.style == "format" or self.style == "pyformat":
            d = self._data[nm]
            code = ""
            if isinstance(d, int):
                code = "d"
            elif isinstance(d, float):
                code = "f"
            elif isinstance(d, str):
                code = "s"
            else:
                raise TypeError(
                    "Can't use column of type {} in '{}' style for column {}".format(
                        type(d), self.style, nm
                    )
                )

            if self.style == "format":
                self.args.append(d)
                return f"%{code}"
            else:
                return f"%({nm}){code}"
        else:
            raise ValueError("Unknown style '{}'".format(self.style))


class DatabaseLogger(Logger, ABC):
    """Generic class to log to any Python Database API database."""

    @property
    @abstractmethod
    def dbapi(self):
        """Return the module associated with this database."""
        pass

    @property
    @abstractmethod
    def database(self):
        """Get the database associated with this logger."""
        pass

    @abstractmethod
    def transaction(self):
        """Context manager to create a database transaction.

        Returns the database object (should be same as one returned by self.database)
        """
        pass

    @contextlib.contextmanager
    def cursor(self):
        """Context manager to create a cursor."""
        cur = self.database.cursor()
        try:
            yield cur
        finally:
            cur.close()

    @property
    def schema_name(self):
        """Get the name of the schema to perform transactions in.

        Return blank for no schema.
        """
        return ""

    def get_schema_version(self):
        """Get the schema version for the database."""
        with self.cursor() as cursor:
            try:
                cursor.execute(
                    f"""SELECT version FROM {self.schema_name}migration
                        ORDER BY version DESC"""
                )
                f = cursor.fetchone()
                if f is None:
                    return 0
                else:
                    return int(f[0])
            except self.dbapi.Error:
                return 0

    def _setup_database(self):
        """Set up the database for the first time, or upgrade it."""
        v = self.get_schema_version()

        def upgrade(vernum):
            nonlocal v
            values = _DbApiValues(self.dbapi, version=vernum)
            cursor.execute(f"""
              INSERT INTO {self.schema_name}migration VALUES ({values.version})
            """, values.params)
            v = vernum

        with self.cursor() as cursor:
            if v < 1:
                cursor.execute(f"""
                  CREATE TABLE {self.schema_name}migration (
                    version INT PRIMARY KEY
                  )""")
                cursor.execute(f"""
                  CREATE TABLE {self.schema_name}event (
                    id TEXT PRIMARY KEY, -- UUID
                    event_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    graph TEXT,
                    instance TEXT,
                    op TEXT,
                    data TEXT
                  )""")
                upgrade(1)

    def save_entry(self, log_entry: LogEntry):
        """Save the entry to the database."""
        id = str(uuid.uuid4())
        with self.transaction():
            with self.cursor() as c:
                values = _DbApiValues(
                    self.dbapi,
                    id=id,
                    event_type=log_entry.event_type.value,
                    timestamp=log_entry.timestamp,
                    graph=log_entry.graph,
                    instance=log_entry.instance,
                    op=log_entry.op,
                    data=dumps_json(log_entry.data),
                )
                c.execute(
                    f"""
                  INSERT INTO {self.schema_name}event(id, event_type, timestamp,
                                                      graph, instance, op, data)
                  VALUES ({values.id}, {values.event_type},
                          {values.timestamp}, {values.graph},
                          {values.instance}, {values.op},
                          {values.data})
                """, values.params)
