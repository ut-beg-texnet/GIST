import shapefile
import os
import sys

def inspect_shapefile(file_path):
    """
    Inspects a shapefile and prints its metadata, schema, and sample records.
    
    Args:
        file_path (str): Path to the shapefile (can be .shp, .dbf, or the base name).
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path) and not os.path.exists(file_path + ".shp"):
            print(f"Error: File '{file_path}' not found.")
            return

        # Load the shapefile
        sf = shapefile.Reader(file_path)
        print(f"Inspecting: {file_path}")
        print("-" * 30)
        
        # Metadata
        print(f"Shape Type: {sf.shapeTypeName}")
        print(f"Number of Records: {len(sf)}")
        print(f"Bounding Box: {sf.bbox}")
        
        # Projection (PRJ file)
        # Try to find the .prj file by replacing extension or appending it
        base_path = os.path.splitext(file_path)[0]
        prj_path = base_path + ".prj"
        if os.path.exists(prj_path):
            with open(prj_path, 'r') as prj:
                print(f"Projection (PRJ): {prj.read().strip()}")
        else:
            print("Projection (PRJ): Not found")
            
        # Schema (Fields)
        print("\nSchema (Fields):")
        # sf.fields returns [('DeletionFlag', 'C', 1, 0), ('FieldName', 'Type', Length, Decimal)...]
        # The first field is always 'DeletionFlag', so we skip it.
        for field in sf.fields[1:]:
            print(f" - {field[0]}: Type={field[1]}, Length={field[2]}")
            
        # Sample Data
        print("\nSample Records (First 5):")
        records = sf.records()
        shapes = sf.shapes()
        for i, (record, shape) in enumerate(zip(records[:5], shapes[:5])):
            # as_dict() is a convenient way to see field names and values
            print(f" Record {i}: {record.as_dict()}")
            # Print coordinates
            if len(shape.points) > 0:
                print(f"  - Point (X, Y): {shape.points[0]}")
            
            # Print Z coordinate if available (POINTZ, POLYGONZ, etc.)
            if hasattr(shape, 'z') and len(shape.z) > 0:
                print(f"  - Z Coordinate (Elevation/Depth): {shape.z[0]}")
            
            # Print M (Measure) if available
            if hasattr(shape, 'm') and len(shape.m) > 0:
                print(f"  - M Coordinate (Measure): {shape.m[0]}")
            
    except shapefile.ShapefileException as e:
        print(f"Shapefile Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_shapefile.py <path_to_shapefile_base>")
        print("Example: python inspect_shapefile.py my_data/points")
    else:
        inspect_shapefile(sys.argv[1])
