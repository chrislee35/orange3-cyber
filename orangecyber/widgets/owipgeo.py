#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy
import ipaddress
from dataclasses import dataclass
from contextlib import contextmanager

from Orange.data import StringVariable, Table, Domain, Variable, ContinuousVariable
from orangewidget.utils import enum_as_int
from orangewidget.settings import ContextSetting, Setting
from orangewidget.widget import Msg
from Orange.widgets import widget, gui
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState

from functools import partial
from typing import Any, Dict, Optional, Set

import geoip2.database

from AnyQt.QtCore import (
    QAbstractTableModel,
    Qt,
)
from AnyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGridLayout,
    QHeaderView,
    QTableView,
)

@dataclass
class Result:
    result_table: Optional[Table] = None

class Enrichment:
    def __init__(self, function):
        self.function = function
        self.types = {StringVariable}

ENRICHMENTS = {
    "Latitude": Enrichment("city.location.latitude"),
    "Longitude": Enrichment("city.location.longitude"),
    "Country": Enrichment("city.country.name"),
    "Country ISO": Enrichment("city.country.iso_code"),
    "Subdivision": Enrichment("city.subdivisions.most_specific.name"),
    "City": Enrichment("city.city.name"),
    "Postal": Enrichment("city.postal.code"),
    "Network": Enrichment("city.traits.network"),
    "ASN": Enrichment("asn.autonomous_system_number"),
    "ASN Org": Enrichment("asn.autonomous_system_organization")
}

ENRICHMENTS_ORD = list(ENRICHMENTS)

class TabColumn:
    attribute = 0
    enrichments = 1

TABLE_COLUMN_NAMES = ["Attributes", "Enrichments"]

class VarTableModel(QAbstractTableModel):
    def __init__(self, parent: "OWIPEnrichment", *args):
        super().__init__(*args)
        self.domain = None
        self.parent = parent

    def set_domain(self, domain: Domain) -> None:
        """
        Reset the table view to new domain
        """
        self.domain = domain
        self.modelReset.emit()

    def update_enrichments(self, attribute: str) -> None:
        """
        Reset the enrichments values in the table for the attribute
        """
        index = self.domain.index(attribute)
        if index < 0:
            # indices of metas are negative: first meta -1, second meta -2, ...
            index = len(self.domain.variables) - 1 - index
        index = self.index(index, 1)
        self.dataChanged.emit(index, index)

    def rowCount(self, parent=None) -> int:
        return (
            0
            if self.domain is None or (parent is not None and parent.isValid())
            else len(self.domain.variables) + len(self.domain.metas)
        )

    @staticmethod
    def columnCount(parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(TABLE_COLUMN_NAMES)

    def data(self, index, role=Qt.DisplayRole) -> Any:
        row, col = index.row(), index.column()
        val = (self.domain.variables + self.domain.metas)[row]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == TabColumn.attribute:
                return str(val)
            else:  # col == TabColumn.aggregations
                # plot first two aggregations comma separated and write n more
                # for others
                enrs = sorted(
                    self.parent.enrichments.get(val, []), key=ENRICHMENTS_ORD.index
                )
                n_more = "" if len(enrs) <= 3 else f" and {len(enrs) - 3} more"
                return ", ".join(enrs[:3]) + n_more
        elif role == Qt.DecorationRole and col == TabColumn.attribute:
            return gui.attributeIconDict[val]
        return None

    def headerData(self, i, orientation, role=Qt.DisplayRole) -> str:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole and i < 2:
            return TABLE_COLUMN_NAMES[i]
        return super().headerData(i, orientation, role)


class CheckBox(QCheckBox):
    def __init__(self, text, parent):
        super().__init__(text)
        self.parent: OWIPEnrichment = parent

    def nextCheckState(self) -> None:
        """
        Custom behaviour for switching between steps. It is required since
        sometimes user will select different types of attributes at the same
        time. In this case we step between unchecked, partially checked and
        checked or just between unchecked and checked - depending on situation
        """
        if self.checkState() == Qt.Checked:
            # if checked always uncheck
            self.setCheckState(Qt.Unchecked)
        else:
            enr = self.text()
            selected_attrs = self.parent.get_selected_attributes()
            types = set(type(attr) for attr in selected_attrs)
            can_be_applied_all = types <= ENRICHMENTS[enr].types

            # true if aggregation applied to all attributes that can be
            # aggregated with selected aggregation
            applied_all = all(
                type(attr) not in ENRICHMENTS[enr].types
                or enr in self.parent.enrichments[attr]
                for attr in selected_attrs
            )
            if self.checkState() == Qt.PartiallyChecked:
                # if partially check: 1) check if agg can be applied to all
                # 2) else uncheck if agg already applied to all
                # 3) else leve partially checked to apply to all that can be aggregated
                if can_be_applied_all:
                    self.setCheckState(Qt.Checked)
                elif applied_all:
                    self.setCheckState(Qt.Unchecked)
                else:
                    self.setCheckState(Qt.PartiallyChecked)
                    # since checkbox state stay same signal is not emitted
                    # automatically but we need a callback call so we emit it
                    self.stateChanged.emit(enum_as_int(Qt.PartiallyChecked))
            else:  # self.checkState() == Qt.Unchecked
                # if unchecked: check if all can be checked else partially check
                self.setCheckState(
                    Qt.Checked if can_be_applied_all else Qt.PartiallyChecked
                )

@contextmanager
def block_signals(widget):
    widget.blockSignals(True)
    try:
        yield
    finally:
        widget.blockSignals(False)


class OWIPEnrichment(widget.OWWidget, ConcurrentWidgetMixin):
    name = "Enrich IP"
    description = "Adds lat/long/city/state/country/ASN/netblock features for IP addresses"
    icon = "icons/ipgeo.svg"
    category = "Cyber"
    keywords = "geo, cyber"
    priority = 9910

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Enriched Data", Table)

    class Error(OWWidget.Error):
        unexpected_error = Msg("{}")

    want_main_area = False
    settingsHandler = DomainContextHandler()
    enrichments: Dict[Variable, Set[str]] = ContextSetting({})
    auto_commit: bool = Setting(True)

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        
        self.data = None
        self.result = None

        self.enr_table_model = VarTableModel(self)
        self.enr_checkboxes = {}

        self.__init_control_area()
        
    def __init_control_area(self) -> None:
        """Init all controls in the main area"""
        # aggregation table
        self.enr_table_view = tableview = QTableView()
        tableview.setModel(self.enr_table_model)
        tableview.setSelectionBehavior(QAbstractItemView.SelectRows)
        tableview.selectionModel().selectionChanged.connect(self.__rows_selected)
        tableview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        vbox = gui.vBox(self.controlArea, " ")
        vbox.layout().addWidget(tableview)

        # aggregations checkboxes
        grid_layout = QGridLayout()
        gui.widgetBox(self.controlArea, orientation=grid_layout, box="Aggregations")

        col = 0
        row = 0
        break_rows = (6, 6, 99)
        for enr in ENRICHMENTS:
            self.enr_checkboxes[enr] = cb = CheckBox(enr, self)
            cb.setDisabled(True)
            cb.stateChanged.connect(partial(self.__enrichment_changed, enr))
            grid_layout.addWidget(cb, row, col)
            row += 1
            if row == break_rows[col]:
                row = 0
                col += 1
                
        gui.auto_send(self.buttonsArea, self, "auto_commit")
        
    @Inputs.data
    def set_data(self, data: Table) -> None:
        self.closeContext()
        self.data = data

        # reset states
        self.cancel()
        self.result = Result()
        self.Outputs.data.send(None)
        self.enrichments = (
            {
                attr: set()
                for attr in data.domain.variables + data.domain.metas
            }
            if data
            else {}
        )
        default_enrichments = self.enrichments.copy()

        self.openContext(self.data)

        # restore aggregations
        self.enrichments.update({k: v for k, v in default_enrichments.items()
                                  if k not in self.enrichments})

        # update selections in widgets and re-plot
        self.enr_table_model.set_domain(data.domain if data else None)

        self.commit.now()
        
    def __enrichment_changed(self, enr: str) -> None:
        """
        Callback for enrichment change; update enrichment dictionary and call
        commit
        """
        selected_attrs = self.get_selected_attributes()
        for attr in selected_attrs:
            if self.enr_checkboxes[enr].isChecked() and self.__enrichment_compatible(
                enr, attr
            ):
                self.enrichments[attr].add(enr)
            else:
                self.enrichments[attr].discard(enr)
            self.enr_table_model.update_enrichments(attr)
        self.commit.deferred()

    def __rows_selected(self) -> None:
        """Callback for table selection change; update checkboxes"""
        selected_attrs = self.get_selected_attributes()

        types = {type(attr) for attr in selected_attrs}
        active_enrichments = [self.enrichments[attr] for attr in selected_attrs]
        for enr, cb in self.enr_checkboxes.items():
            cb.setDisabled(not types & ENRICHMENTS[enr].types)

            activated = {enr in a for a in active_enrichments}
            with block_signals(cb):
                # check if enrichment active for all selected attributes,
                # partially check if active for some else uncheck
                cb.setCheckState(
                    Qt.Checked
                    if activated == {True}
                    else (Qt.Unchecked if activated == {False} else Qt.PartiallyChecked)
                )

    @staticmethod
    def __enrichment_compatible(agg, attr):
        """Check a compatibility of enrichment with the variable"""
        return type(attr) in ENRICHMENTS[agg].types

    def get_selected_attributes(self):
        """Get select attributes in the table"""
        selection_model = self.enr_table_view.selectionModel()
        sel_rows = selection_model.selectedRows()
        vars_ = self.data.domain.variables + self.data.domain.metas
        return [vars_[index.row()] for index in sel_rows]

    @gui.deferred
    def commit(self) -> None:
        self.Error.clear()
        self.Warning.clear()
        if self.data:
            self.start(_run, self.data, self.enrichments, self.result)

    def on_done(self, result: Table) -> None:
        self.result = result
        self.Outputs.data.send(result.result_table)

    def on_partial_result(self, result: Table) -> None:
        # store result in case the task is canceled and on_done is not called
        self.result = result

    def on_exception(self, ex: Exception):
        self.Error.unexpected_error(str(ex))
        
        
class GeoIP2Manager:
    city = None
    asn = None
    
    def __init__(self, city_mmdb=None, asn_mmdb=None):
        GeoIP2Manager.city = geoip2.database.Reader(city_mmdb)
        GeoIP2Manager.asn = geoip2.database.Reader(asn_mmdb)
    
def _run(
    data: Table,
    enrichments: Dict[Variable, Set[str]],
    result: Result,
    state: TaskState,
) -> Table:
    def progress(part):
        state.set_progress_value(part * 100)
        if state.is_interruption_requested():
            raise Exception

    state.set_status("Enriching")

    if len(enrichments) == 0: return data
    
    #domain = data.domain
    enrichment_variables = []
    for var, enrs in enrichments.items():
        for e in list(enrs):
            if e in ('Latitude', 'Longitude'):
                enrichment_variables.append(ContinuousVariable(str(var)+" - "+e))
            else:
                enrichment_variables.append(StringVariable(str(var)+" - "+e))    

    total_actions = len(enrichment_variables) * data.approx_len()
    count_actions = 0
    
    values = []
    for var in enrichments.keys():
        col = data.get_column(var)
        need_asn = enrichments[var].intersection({'ASN', 'ASN Org'}) != set()
        need_city = (enrichments[var] - {'ASN', 'ASN Org'}) != set()
        
        for ip in col:
            row = []
            city = asn = None
            try:
                # check if IP is valid
                ipaddress.IPv4Address(ip)
                
                # only lookup city if there is a need
                if need_city: city = GeoIP2Manager.city.city(ip)
                # only lookup asn if there is a need
                if need_asn: asn = GeoIP2Manager.asn.asn(ip)
            except Exception:
                pass
            
            for enrs in list(enrichments[var]):
                if not city and not asn:
                    val = None
                else:
                    val = eval(ENRICHMENTS[enrs].function)
                row.append(val)
                count_actions += 1
            values.append(row)
            progress(count_actions / total_actions)
    
    enrichment_domain = Domain(
        data.domain.attributes,
        data.domain.class_vars,
        data.domain.metas + tuple(enrichment_variables)
    )
    
    orig = data.metas
    enrich = numpy.array(values)
    new_data = numpy.concatenate( (orig, enrich), axis=1 )
    result.result_table = Table.from_numpy(
        enrichment_domain,
        data.X, data.Y, new_data
    )
    
    return result
        
def main(argv=sys.argv):
    
    GeoIP2Manager('/home/chris/Downloads/GeoLite2-City.mmdb', '/home/chris/Downloads/GeoLite2-ASN.mmdb')
    from orangewidget.utils.widgetpreview import WidgetPreview
    data = [ [x] for x in "1.1.1.1,2.2.2.2,3.3.3.3,4.4.4.4,jerry".split(",") ]
    varibles = [ StringVariable("IP Address") ]
    domain = Domain([], [], varibles)    
    dataset = Table.from_list(domain, data)
    WidgetPreview(OWIPEnrichment).run(dataset)

    return 0

if __name__ == "__main__":
    sys.exit(main())