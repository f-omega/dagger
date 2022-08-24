"""SQLite logger."""

from .db import DatabaseLogger

import sqlite3
import contextlib
import threading


class SqliteLogger(DatabaseLogger):
    """A Logger that writes to a SQLite database.

    Useful for testing, or local execution
    """

    def __init__(self, filepath):
        """Initialize the database logger from a file path."""
        self.filepath = filepath
        self.by_thread = threading.local()
        self.by_thread.connection = None

        with self.transaction():
            super()._setup_database()

    @property
    def dbapi(self):
        """Return the sqlite3 module."""
        return sqlite3

    @property
    def database(self):
        """Return the sqlite connection."""
        if not hasattr(self.by_thread, 'connection') or \
           self.by_thread.connection is None:
            self.by_thread.connection = sqlite3.connect(self.filepath)

        return self.by_thread.connection

    @contextlib.contextmanager
    def transaction(self):
        """Run a transaction."""
        with self.database:
            yield self.database

