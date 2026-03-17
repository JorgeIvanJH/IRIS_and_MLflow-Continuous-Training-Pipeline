import pyodbc

# Configuration
server = 'localhost'
port = '1972' # Standard InterSystems superserver port
database = 'USER'
username = 'SuperUser'
password = 'SYS'


# Connection string using the specific InterSystems ODBC driver
# Note: Ensure the driver name exactly matches what is installed on your client
# Updated with the correct driver name from your system
connection_string = (
    f"DRIVER={{InterSystems IRIS ODBC35}};"
    f"SERVER={server};"
    f"PORT={port};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password}"
)

query = "SELECT * FROM MLpipeline.NoShowsAppointments"

try:
    # Establish connection
    cnxn = pyodbc.connect(connection_string)
    cursor = cnxn.cursor()
    
    # Execute query
    cursor.execute(query)
    
    # Fetch and display results
    for row in cursor.fetchall():
        print(row)
        
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'cnxn' in locals():
        cnxn.close()
