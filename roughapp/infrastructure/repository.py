"""
Repository for saving and loading decision tables using sqlite3.

sqlite3 was chosen because it requires no server setup and a single file is
sufficient for the scope of this project. The repository pattern ensures that
the service layer never writes SQL directly — all storage details stay here.

This follows the Dependency Inversion Principle: the service depends on
TableRepository, not on sqlite3 directly, which means changing the storage
implementation does not affect the rest of the application.
"""
