Class MLpipeline.FeatureStore Extends %RegisteredObject
{

// Example of a constant parameter that never changes

Parameter ConstVar = "CONSTANT VAR FOR THE PROJECT";

Method %OnNew() As %Status [ Private ]
{
    do ##class(%SYS.System).WriteToConsoleLog("Created instance of MLpipeline.FeatureStore with ConstVar: "_..#ConstVar, 0, 0)
    Quit $$$OK
}

/*
Populates an IRIS table data from CSV file at csvpath. Automatically detects and converts date columns. 
e.g:
    set FS = ##class(MLpipeline.FeatureStore).%New()
    set status = FS.UploadCSVtoIRIS("/dur/data/point_samples.csv","USER", "MLpipeline","PointSamples")
*/
Method UploadCSVtoIRIS(csvpath As %String, namespace As %String, package As %String, tablename As %String) As %Status [ Language = python ]
{
    from sqlalchemy import create_engine, text
    import pandas as pd
    import iris
    
    try:
        iris._SYS.System.WriteToConsoleLog(f"Starting CSV upload to IRIS: {csvpath} into {namespace}.{package}.{tablename}", 0, 0)
        engine = create_engine(f"iris+emb:///{namespace}")

        create_table_sql = f"""
        CREATE TABLE {package}.{tablename} (
            ID INTEGER PRIMARY KEY IDENTITY(1,1),
            x DOUBLE,
            y DOUBLE,
            label VARCHAR(50),
            datetime DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        with engine.connect() as conn:
            # Check if table exists
            check_sql = text("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = :sch AND TABLE_NAME = :tab")
            result = conn.execute(check_sql, {"sch": package, "tab": tablename}).fetchone()
            
            if not result:
                conn.execute(text(create_table_sql))
                conn.commit()

        df = pd.read_csv(csvpath)

        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().astype(str).head(20)
            try:
                parsed = pd.to_datetime(sample, errors="raise") # just to validate
                df[col] = pd.to_datetime(df[col], errors="coerce") # actual conversion
            except Exception:
                continue
        df.to_sql(tablename, engine, schema=package, if_exists='append', index=False, method='multi')
        iris._SYS.System.WriteToConsoleLog("Successfully uploaded CSV to IRIS", 0, 0)
        return True

    except Exception as e:
        print(f"Error uploading CSV to IRIS: {e}")
        iris._SYS.System.WriteToConsoleLog(f"Error uploading CSV to IRIS: {e}", 0, 2)
        return False
}

/*
Extracts data from IRIS database using SQL query constructed from the input parameters.
e.g:
   set FS = ##class(MLpipeline.FeatureStore).%New()
   set rs = FS.DataExtraction("MLpipeline", "PointSamples", "*", "datetime > '2026-02-26 17:50:00'")
*/
Method DataExtraction(schema As %String, tablename As %String, columns As %String = "*", filters As %String = "") As %SQL.StatementResult
{
    set query = "SELECT " _ columns _ " FROM " _ schema _ "." _ tablename
    if (filters '= "") set query = query _ " WHERE " _ filters

    set tStatement = ##class(%SQL.Statement).%New()
    set tStatus = tStatement.%Prepare(query)
    
    // Handle Preparation Errors (e.g., syntax errors)
    if $$$ISERR(tStatus) {
        do ##class(%SYS.System).WriteToConsoleLog("SQL Prepare Failed: " _ query, 0, 2)
        quit ""
    }

    set rs = tStatement.%Execute()

    // Handle Execution Errors
    if (rs.%SQLCODE < 0) {
        do ##class(%SYS.System).WriteToConsoleLog("SQL Execution Error: " _ rs.%Message, 0, 2)
        quit rs // Return the result set anyway; Python will see the error code
    }
    
    do ##class(%SYS.System).WriteToConsoleLog("Executed query: " _ query, 0, 0)
    
    return rs
}

/*
Creates a Predictions table in IRIS to store inference results, with a foreign key relationship to the PointSamples table.
e.g:
    set FS = ##class(MLpipeline.FeatureStore).%New()
    set status = FS.CreatePredictionsTable("USER", "MLpipeline", "Predictions")
*/
Method CreatePredictionsTable(namespace As %String = "USER", package As %String = "MLpipeline", tablename As %String = "Predictions") As %Status [ Language = python ]
{
    from sqlalchemy import create_engine, text
    import iris

    try:
        engine = create_engine(f"iris+emb:///{namespace}")
        
        create_table_sql = f"""
        CREATE TABLE {package}.{tablename} (
            ID INTEGER PRIMARY KEY IDENTITY(1,1),
            PointsIDForeignKey INTEGER,
            YPred DOUBLE,
            Modelrunid VARCHAR(100),
            MAE DOUBLE,
            InferenceTime DOUBLE,
            datetime DATETIME DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_PointSamples FOREIGN KEY (PointsIDForeignKey) 
            REFERENCES {package}.PointSamples(ID)
        )
        """

        with engine.connect() as conn:
            # Check if table already exists to avoid errors
            check_sql = text(f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = :sch AND TABLE_NAME = :tbl")
            result = conn.execute(check_sql, {"sch": package, "tbl": tablename}).fetchone()
            
            if not result:
                conn.execute(text(create_table_sql))
                conn.commit()
                print(f"Table {package}.{tablename} created successfully.")
            else:
                print(f"Table {package}.{tablename} already exists.")
        
        return True

    except Exception as e:
        print(f"Error creating {package}.{tablename} table: {e}")
        iris._SYS.System.WriteToConsoleLog(f"Error creating {package}.{tablename} table: {e}", 0, 2)
        return False
}

}
