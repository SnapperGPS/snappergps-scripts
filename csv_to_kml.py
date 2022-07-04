# -*- coding: utf-8 -*-
"""
Convert CSV file with GNSS positions into KML files:

Expects CSV files from clean_csv.py

@author: Jonas Beuchert
"""
import glob
import numpy as np

# Path to files goes here
files = glob.glob("*_reduced.csv")

for file in files:

    print(f"Process {file}.")

    lats, lons = np.loadtxt(file,
                            delimiter=',',
                            skiprows=1,
                            usecols=[1, 2],
                            unpack=True,
                            dtype="str")

    with open(f"{file[:-4]}.kml", "w") as file_pointer:
        file_pointer.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
    <Document id="root_doc">
        <Schema name="{file[:-4]}_KML" id="{file[:-4]}_KML">
            <SimpleField name="descriptio" type="string"></SimpleField>
            <SimpleField name="timestamp" type="string"></SimpleField>
            <SimpleField name="begin" type="string"></SimpleField>
            <SimpleField name="end" type="string"></SimpleField>
            <SimpleField name="altitudeMo" type="string"></SimpleField>
            <SimpleField name="tessellate" type="float"></SimpleField>
            <SimpleField name="extrude" type="float"></SimpleField>
            <SimpleField name="visibility" type="float"></SimpleField>
            <SimpleField name="drawOrder" type="float"></SimpleField>
            <SimpleField name="icon" type="string"></SimpleField>
            <SimpleField name="snippet" type="string"></SimpleField>
        </Schema>
        <Folder>
            <name>{file[:-4]}_KML</name>
            <Placemark>
                <name>SnapperGPS data</name>
                <Style>
                    <LineStyle>
                        <color>ff0000ff</color>
                    </LineStyle>
                    <PolyStyle>
                        <fill>0</fill>
                    </PolyStyle>
                </Style>
                <ExtendedData>
                    <SchemaData schemaUrl="#{file[:-4]}_KML">
                        <SimpleData name="descriptio">&amp;nbsp;</SimpleData>
                        <SimpleData name="altitudeMo">clampToGround</SimpleData>
                        <SimpleData name="tessellate">1</SimpleData>
                        <SimpleData name="extrude">0</SimpleData>
                        <SimpleData name="visibility">-1</SimpleData>
                    </SchemaData>
                </ExtendedData>
                <MultiGeometry>
                    <LineString>
                        <coordinates>
""")
        for lat, lon in zip(lats, lons):
            file_pointer.write(f"""                            {lon},{lat}
""")
        file_pointer.write("""                        </coordinates>
                    </LineString>
                </MultiGeometry>
            </Placemark>
        </Folder>
    </Document>
</kml>
""")

        print()
