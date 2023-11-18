from os import path
from traceback import format_exception_only

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QStyle, QSizePolicy, QFileDialog

#from Orange.util import get_entry_point
from Orange.data import Table, Domain, StringVariable, DiscreteVariable
from Orange.data.util import get_unique_names
from Orange.widgets import gui, settings
from Orange.widgets.settings import ContextHandler
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from orangecontrib.network.network import Network
from orangecontrib.network.network.base import DirectedEdges

import stix2
import scipy.sparse as sp

class StixContextHandler(ContextHandler):
    def new_context(self, useful_vars):
        context = super().new_context()
        context.useful_vars = {var.name for var in useful_vars}
        context.label_variable = 'name'
        return context

    # noinspection PyMethodOverriding
    def match(self, context, useful_vars):
        useful_vars = {var.name for var in useful_vars}
        if context.useful_vars == useful_vars:
            return self.PERFECT_MATCH
        # context.label_variable can also be None; this would always match,
        # so ignore it
        elif context.label_variable in useful_vars:
            return self.MATCH
        else:
            return self.NO_MATCH

    def settings_from_widget(self, widget, *_):
        context = widget.current_context
        if context is not None:
            context.label_variable = \
                widget.label_variable and widget.label_variable.name

    def settings_to_widget(self, widget, useful_vars):
        context = widget.current_context
        widget.label_variable = None
        if context.label_variable is not None:
            for var in useful_vars:
                if var.name == context.label_variable:
                    widget.label_variable = var
                    break


class OWStix(OWWidget):
    name = "STIX File"
    description = "Read in a STIX 2.1 file"
    icon = "icons/stix-attack_pattern.svg"
    priority = 6410

    class Inputs:
        items = Input("Items", Table)

    class Outputs:
        network = Output("Network", Network)
        items = Output("Items", Table)

    settingsHandler = StixContextHandler()
    label_variable: StringVariable = settings.ContextSetting('name')
    recentFiles = settings.Setting([])

    class Information(OWWidget.Information):
        auto_annotation = Msg(
            'Nodes annotated with data from file with the same name')
        suggest_annotation = Msg(
            'Add optional data input to annotate nodes')

    class Error(OWWidget.Error):
        io_error = Msg('Error reading file "{}"\n{}')
        error_parsing_file = Msg('Error reading file "{}"')
        auto_data_failed = Msg(
            "Attempt to read {} failed\n"
            "The widget tried to annotated nodes with data from\n"
            "a file with the same name.")
        mismatched_lengths = Msg(
            "Data size does not match the number of nodes.\n"
            "Select a data column whose values can be matched with network "
            "labels")

    want_main_area = False
    mainArea_width_height_ratio = None

    def __init__(self):
        super().__init__()

        self.network = None
        self.auto_data = None
        self.original_nodes = None
        self.data = None
        self.stix_index = 0
        self.label_variable = 'name'

        hb = gui.widgetBox(self.controlArea, orientation=Qt.Horizontal)
        self.filecombo = gui.comboBox(
            hb, self, "stix_index", callback=self.select_stix_file,
            minimumWidth=250)
        gui.button(
            hb, self, '...', callback=self.browse_stix_file, disabled=0,
            icon=self.style().standardIcon(QStyle.SP_DirOpenIcon),
            sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Fixed))
        gui.button(
            hb, self, 'Reload', callback=self.reload,
            icon=self.style().standardIcon(QStyle.SP_BrowserReload),
            sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Fixed))


        self.reload()

    @Inputs.items
    def set_data(self, data):
        self.data = data
        self.update_label_combo()
        self.send_output()

    def populate_comboboxes(self):
        self.filecombo.clear()
        for file in self.recentFiles or ("(None)",):
            self.filecombo.addItem(path.basename(file))
        self.filecombo.addItem("Browse documentation STIX bundles...")
        self.filecombo.updateGeometry()

    def browse_stix_file(self, browse_demos=False):
        """user pressed the '...' button to manually select a file to load"""
        startfile = self.recentFiles[0] if self.recentFiles else '.'

        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open a Stix File', startfile,
            ';;'.join(("STIX files (*.json)",)))
        if not filename:
            return False

        if filename in self.recentFiles:
            self.recentFiles.remove(filename)
        self.recentFiles.insert(0, filename)

        self.populate_comboboxes()
        self.stix_index = 0
        self.select_stix_file()
        return True

    def reload(self):
        if self.recentFiles:
            self.select_stix_file()

    def select_stix_file(self):
        """user selected a STIX file from the combo box"""
        if self.stix_index > len(self.recentFiles) - 1:
            if not self.browse_stix_file(True):
                return  # Cancelled
        elif self.stix_index:
            self.recentFiles.insert(0, self.recentFiles.pop(self.stix_index))
            self.stix_index = 0
            self.populate_comboboxes()
        if self.recentFiles:
            self.open_stix_file(self.recentFiles[0])

    def open_stix_file(self, filename):
        """Read stix from file."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        self.network = None
        self.original_nodes = None
        try:
            self.network = self.read_stix(filename)
        except OSError as err:
            self.Error.io_error(
                filename,
                "".join(format_exception_only(type(err), err)).rstrip())
        except Exception:  # pylint: disable=broad-except
            self.Error.error_parsing_file(filename)
        else:
            self.original_nodes = self.network.nodes
            #self.read_auto_data(filename)
        self.update_label_combo()
        self.send_output()
        
    def read_stix(self, filename):
        try:
            with open(filename, 'r', encoding='UTF-8') as fh:
                bundle = stix2.parse(fh, allow_custom=True)
                objects = [x for x in bundle.objects if x.type != 'relationship']
                relationships = [x for x in bundle.objects if x.type == 'relationship']
                labels = [ x.name for x in objects]
                id_idx = {}
                num_objs = len(objects)
                
                self.stix_types = set([x.type for x in objects])
                
                populated_fields = set()
                
                for i, obj in enumerate(objects):
                    id_idx[obj.id] = i
                    populated_fields.update(set(obj.properties_populated()))
                    if 'object_refs' in obj:
                        for target_ref in obj.object_refs:
                            # you can't have a relationship to a relationship
                            if 'relationship' in target_ref: continue
                            relationships.append(stix2.Relationship(source_ref=obj.id, relationship_type='refers-to', target_ref=target_ref))
                      
                populated_fields = list(populated_fields)
                rows = []
                for i, obj in enumerate(objects):
                    row = []
                    for field in populated_fields:
                        v = obj.get(field)
                        if type(v) == list:
                            row.append(",".join([str(x) for x in v]))
                        elif v is None:
                            row.append(None)
                        else:
                            row.append(str(v))
                    rows.append(row)
    
                num_rels = len(relationships)
                edge_srcs = np.zeros( (num_rels), dtype='int' )
                edge_dsts = np.zeros( (num_rels), dtype='int' )
                edge_vals = np.ones( (num_rels), dtype='float' )
                edge_labels = []
                            
                for i, rel in enumerate(relationships):
                    src_i = id_idx.get(rel.source_ref)
                    dst_i = id_idx.get(rel.target_ref)
                    label = rel.relationship_type
                    if None in (src_i, dst_i, label): continue
                    edge_srcs[i] = src_i
                    edge_dsts[i] = dst_i
                    edge_labels.append(label)
                    
                edges = sp.coo_matrix(
                    (edge_vals, (edge_srcs, edge_dsts)),
                    shape=(num_objs, num_objs)
                )
                
                edges = DirectedEdges(edges, np.array(edge_labels))
    
                coordinates = np.zeros((num_objs, 2))
                network = Network(labels, edges, bundle.id or "", coordinates)
    
                metas = []
                for field in populated_fields:
                    if field == 'type':
                        metas.append(DiscreteVariable("type", values=self.stix_types)),
                    else:
                        metas.append(StringVariable(field))
                        
                self.auto_data_domain = Domain([], [], metas)
                self.auto_data = Table.from_list(self.auto_data_domain, rows)
     
                return network
        except Exception as e:
            print(e)

    def update_label_combo(self):
        self.closeContext()
        data = self.data if self.data is not None else self.auto_data
        if self.network is None or data is None:
            pass
        else:
            self.label_variable = 'name'
        self.set_network_nodes()

    def _vars_for_label(self, data: Table):
        vars_and_overs = []
        original_nodes = set(self.original_nodes)
        for var in data.domain.metas:
            if not isinstance(var, StringVariable):
                continue
            values= data.get_column(var)
            values = values[values != ""]
            set_values = set(values)
            if len(values) != len(set_values) \
                    or not original_nodes <= set_values:
                continue
            vars_and_overs.append((len(set_values - original_nodes), var))
        if not vars_and_overs:
            return None, []
        _, best_var = min(vars_and_overs)
        useful_string_vars = [var for _, var in vars_and_overs]
        return best_var, useful_string_vars

    def send_output(self):
        if self.network is None:
            self.Outputs.network.send(None)
            self.Outputs.items.send(None)
        else:
            self.Outputs.network.send(self.network)
            self.Outputs.items.send(self.network.nodes)

    def set_network_nodes(self):
        self.Error.mismatched_lengths.clear()
        self.Information.auto_annotation.clear()
        self.Information.suggest_annotation.clear()
        if self.network is None:
            return

        data = self.data if self.data is not None else self.auto_data
        if data is None:
            self.Information.suggest_annotation()
        elif self.label_variable is None \
                and len(data) != self.network.number_of_nodes():
            self.Error.mismatched_lengths()
            data = None

        if data is None:
            self.network.nodes = self._label_to_tabel()
        elif self.label_variable is None:
            self.network.nodes = self._combined_data(data)
        else:
            self.network.nodes = self._data_by_labels(data)

    def _data_by_labels(self, data):
        data_col = data.get_column(self.label_variable)
        data_rows = {label: row for row, label in enumerate(data_col)}
        indices = [data_rows[label] for label in self.original_nodes]
        return data[indices]

    def _combined_data(self, source):
        nodes = np.array(self.original_nodes, dtype=str)
        if nodes.ndim != 1:
            return source
        try:
            nums = np.sort(np.array([int(x) for x in nodes]))
        except ValueError:
            pass
        else:
            if np.all(nums[1:] - nums[:-1] == 1):
                return source

        src_dom = source.domain
        label_attr = StringVariable(get_unique_names(src_dom, "node_label"))
        domain = Domain(src_dom.attributes, src_dom.class_vars,
                        src_dom.metas + (label_attr, ))
        data = source.transform(domain)
        with data.unlocked(data.metas):
            data.metas[:, -1] = nodes
        return data

    def _label_to_tabel(self):
        domain = Domain([], [], [
            StringVariable("id"), 
            DiscreteVariable("type", values=self.stix_types), 
            StringVariable("name")
        ])
        n = len(self.original_nodes)
        data = Table.from_numpy(
            domain, np.empty((n, 0)), np.empty((n, 0)),
            np.array([(x.id, x.type, x.name) for x in self.original_nodes], dtype=str).reshape(-1, 3))
        return data


    def sendReport(self):
        self.reportSettings(
            "STIX file",
            [("File name", self.filecombo.currentText()),
             ("Vertices", self.network.number_of_nodes()),
             ("Directed", gui.YesNo[self.network.edges[0].directed])
             ])


if __name__ == "__main__":
    WidgetPreview(OWStix).run()