Class MLpipeline.DataManager Extends %RegisteredObject
{

/*
Populates an IRIS table data from CSV file at csvpath. Automatically detects and converts date columns. 
e.g:
    do ##class(MLpipeline.DataManager).UploadCSVtoIRIS("/dur/data/point_samples.csv","USER", "MLpipeline","PointSamples")
*/
ClassMethod UploadCSVtoIRIS(csvpath As %String, namespace As %String, package As %String, tablename As %String) As %Status [ Language = python ]
{
    from sqlalchemy import create_engine, text
    import pandas as pd
    import iris
    
    try:
        engine = create_engine(f"iris+emb:///{namespace}")

        create_table_sql = f"""
        CREATE TABLE {package}.{tablename} (
            ID INTEGER PRIMARY KEY IDENTITY(1,1),
            x DOUBLE,
            y DOUBLE,
            label VARCHAR(50),
            datetime DATETIME
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
        return True

    except Exception as e:
        print(f"Error uploading CSV to IRIS: {e}")
        return False
}

}

