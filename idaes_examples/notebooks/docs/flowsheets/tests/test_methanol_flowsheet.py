import pytest

# Import Pyomo libraries
from pyomo.environ import (
    Constraint,
    Objective,
    Var,
    Expression,
    Param,
    ConcreteModel,
    TransformationFactory,
    value,
    maximize,
    units as pyunits,
)
from pyomo.environ import TerminationCondition
from pyomo.network import Arc

# Import IDAES core libraries
from idaes.core import FlowsheetBlock
from idaes.core.solvers import get_solver
from idaes.core.util import scaling as iscale
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state

# Import required models

from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.base.generic_reaction import (
    GenericReactionParameterBlock,
)

from idaes_examples.mod.methanol import (
    methanol_ideal_VLE as thermo_props_VLE,
    methanol_ideal_vapor as thermo_props_vapor,
    methanol_reactions as reaction_props,
)

from idaes.models.unit_models import (
    Feed,
    Mixer,
    Heater,
    Compressor,
    Turbine,
    StoichiometricReactor,
    Flash,
    Product,
)
from idaes.models.unit_models.mixer import MomentumMixingType
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.core import UnitModelCostingBlock
from idaes.models.costing.SSLW import SSLWCosting

from methanol_flowsheet import (
    build_model,
    set_inputs,
    scale_flowsheet,
    initialize_flowsheet,
    add_costing,
    report
)

@pytest.mark.performance
def test_run_methanol_flowsheet():
    m = ConcreteModel()
    solver = get_solver()  # IPOPT
    optarg = {"tol": 1e-6, "max_iter": 500}
    solver.options = optarg
    build_model(m)  # build flowsheet
    set_inputs(m)  # unit and stream specifications
    scale_flowsheet(m)
    initialize_flowsheet(m)  # rigorous initialization scheme
    print("DOF before solve: ", degrees_of_freedom(m))
    print()
    print("Solving initial problem...")
    results = solver.solve(m, tee=True)
    assert results.solver.termination_condition == TerminationCondition.optimal

    add_costing(m)  # re-solve with costing equations
    print()
    results2 = solver.solve(m, tee=True)
    assert results2.solver.termination_condition == TerminationCondition.optimal
    print("Initial solution process results:")
    report(m)  # display initial solution results

    # Set up Optimization Problem (Maximize Revenue)
    # keep process pre-reaction fixed and unfix some post-process specs
    m.fs.R101.conversion.unfix()
    m.fs.R101.conversion_lb = Constraint(expr=m.fs.R101.conversion >= 0.75)
    m.fs.R101.conversion_ub = Constraint(expr=m.fs.R101.conversion <= 0.85)
    m.fs.R101.outlet_temp.deactivate()
    m.fs.R101.outlet_t_lb = Constraint(
        expr=m.fs.R101.control_volume.properties_out[0.0].temperature >= 405 * pyunits.K
    )
    m.fs.R101.outlet_t_ub = Constraint(
        expr=m.fs.R101.control_volume.properties_out[0.0].temperature <= 505 * pyunits.K
    )

    # Optimize turbine work (or delta P)
    m.fs.T101.deltaP.unfix()  # optimize turbine work recovery/pressure drop
    m.fs.T101.outlet_p_lb = Constraint(
        expr=m.fs.T101.outlet.pressure[0] >= 10e5 * pyunits.Pa
    )
    m.fs.T101.outlet_p_ub = Constraint(
        expr=m.fs.T101.outlet.pressure[0] <= 51e5 * 0.8 * pyunits.Pa
    )

    # Optimize Cooler outlet temperature - unfix cooler outlet temperature
    m.fs.H102.outlet_temp.deactivate()
    m.fs.H102.outlet_t_lb = Constraint(
        expr=m.fs.H102.control_volume.properties_out[0.0].temperature
        >= 407.15 * 0.8 * pyunits.K
    )
    m.fs.H102.outlet_t_ub = Constraint(
        expr=m.fs.H102.control_volume.properties_out[0.0].temperature <= 480 * pyunits.K
    )

    m.fs.F101.deltaP.unfix()  # allow pressure change in streams

    m.fs.F101.isothermal = Constraint(
        expr=m.fs.F101.control_volume.properties_out[0].temperature
        == m.fs.F101.control_volume.properties_in[0].temperature
    )

    print()
    print("Solving optimization problem...")
    opt_res = solver.solve(m, tee=True)
    assert opt_res.solver.termination_condition == TerminationCondition.optimal
    print("Optimal solution process results:")
    report(m)