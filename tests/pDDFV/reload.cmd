VERSION 10.02

ext: menu.generic.delete # Begin run()
ext: menu.generic.delete BeginMenuCmd
ext: menu.generic.delete {"menu_version": 1.0, "mousepos": [-1.2345000022229158e-10, -1.2345000022229158e-10], "widget": "list", "target": [["ENS_PART", [1]]], "attribute": 1610612740, "pos": [-1.2345000022229158e-10, -1.2345000022229158e-10, -1.2345000022229158e-10], "menu_extra_info": null, "mode": "Part", "part_selection": [["ENS_PART", [1]]], "pick": 0}
ext: menu.generic.delete EndMenuCmd
ext: menu.generic.delete # End run()
part: select_default
part: modify_begin
part: elt_representation 3D_feature_2D_full
part: modify_end
case: replace 'Case 1' 'Case 1'
case: select Case 1
part: select_default 
part: modify_begin 
part: elt_representation 3D_feature_2D_full 
part: modify_end 
data: binary_files_are big_endian
data: format case
data: shift_time 1.000000 0.000000 0.000000
data: replace /cea/S/dsku/sirius/hal1/home/s3/camierjs/cea/arcane/nabla/pDDFV/output/depouillement/ensight.case
ext: treecmd.treecmds begin
ext: treecmd.treecmds var ''
ext: treecmd.treecmds path 'Scalars'
ext: treecmd.treecmds create_group
ext: treecmd.treecmds begin
ext: treecmd.treecmds var 'cell_cell_th'
ext: treecmd.treecmds path 'Scalars'
ext: treecmd.treecmds reparent_var
ext: treecmd.treecmds begin
ext: treecmd.treecmds var ''
ext: treecmd.treecmds path 'Scalars'
ext: treecmd.treecmds create_group
ext: treecmd.treecmds begin
ext: treecmd.treecmds var 'node_node_th'
ext: treecmd.treecmds path 'Scalars'
ext: treecmd.treecmds reparent_var
part: select_all
part: modify_begin
part: entity_label_node ON
part: modify_end
