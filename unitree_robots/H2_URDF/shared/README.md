# Shared H2 dependencies

`H2_modified_high_damping.xml` and `meshes/` are used by both task 2 and task 3. Scene XMLs include this robot model and set the mesh directory relative to their task directory.

`starter_helper.py` supplies the existing joint control utility and CSV exporter. Runners pass explicit CSV paths so recordings stay with their task. Keep experiment-specific behavior in the task directories.
