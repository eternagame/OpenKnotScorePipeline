from abc import ABC, abstractmethod
import os
import subprocess
import re
import math
from arnie.utils import convert_dotbracket_to_bp_list, convert_bp_list_to_dotbracket
try:
    from arnie.mfe import mfe
    from arnie.bpps import bpps
    from arnie.pk_predictors import pk_predict, pk_predict_from_bpp
except Exception as e:
    # Arnie unfortunately raises an exception on import if no predictors are configured
    # If the user isn't using arnie, we don't want to error, so we'll defer raising an error
    # to when attempting to use these methods
    def arnie_load_failed(*args, **kwargs):
        raise e
    mfe = bpps = pk_predict = pk_predict_from_bpp = arnie_load_failed
from ...substation.scheduler.domain import UtilizedResources

class Predictor(ABC):
    @property
    @abstractmethod
    def uses_experimental_reactivities() -> bool:
        '''
        Whether or not this predictor takes into account the experimental reactivities in
        deriving its prediction. If it does not, we can use cached values for this sequence
        even if the reactivities changed (and also we can run it even if experimental data
        is not available)
        '''
        pass

    @property
    @abstractmethod
    def prediction_names(self) -> list[str]:
        pass

    @abstractmethod
    def run(self, seq: str, reactivities: list[float]) -> dict[str, str]:
        '''
        Run the given predictor, returning a map of prediction names to dot-bracket
        structures. We allow multiple predictions to be returned in case multiple different
        settings can be more efficiently computed in parallel (eg, computing bpps once and then
        running both hungarian and threshknot)
        '''
        pass

    @abstractmethod
    def approximate_resources(self, seq: str) -> UtilizedResources:
        pass

    gpu = False

class ArnieMfePredictor(Predictor):
    uses_experimental_reactivities = False

    def __init__(self, package_name: str, as_name: str, arnie_kwargs: dict = None):
        self.name = as_name
        self.package_name = package_name
        self.arnie_kwargs = arnie_kwargs or {}

    @property
    def prediction_names(self):
        return [self.name]

    def run(self, seq: str, reactivities: list[float]):
        return {
            self.name: mfe(seq, self.package_name, **self.arnie_kwargs)
        }

class ArnieMfeShapePredictor(ArnieMfePredictor):
    uses_experimental_reactivities = True

    def run(self, seq: str, reactivities: list[float]):
        # arnie expects a list of float values; we need to account for the leader region with invalid data,
        # so we don't provide the reactivity data directly stored in the row
        processed_reactivities = [value if value is not None else -1 for value in reactivities]
        return {
            self.name: mfe(seq, self.package_name, shape_signal=processed_reactivities, **self.arnie_kwargs)
        }
    
class ArniePkPredictor(Predictor):
    uses_experimental_reactivities = False

    def __init__(self, package_name: str, as_name: str, arnie_kwargs: dict = None):
        self.name = as_name
        self.package_name = package_name
        self.arnie_kwargs = arnie_kwargs or {}

    @property
    def prediction_names(self):
        return [self.name]

    def run(self, seq: str, reactivities: list[float]):
        return {
            self.name: pk_predict(seq, self.package_name, **self.arnie_kwargs)
        }

class ArniePkFromBppPredictor(Predictor):
    uses_experimental_reactivities = False

    def __init__(self, package_name: str, arnie_bpp_kwargs: dict = None):
        self.package_name = package_name
        self.arnie_bpp_kwargs = arnie_bpp_kwargs or {}
        self.heuristics = []

    def add_heuristic(self, heuristic: str, as_name: str, arnie_heuristic_kwargs: dict = None):
        self.prediction_names.append(as_name)
        self.heuristics.append({
            'heuristic': heuristic,
            'arnie_heuristic_kwargs': arnie_heuristic_kwargs,
            'name': as_name
        })
        return self

    @property
    def prediction_names(self):
        return [heuristic['name'] for heuristic in self.heuristics]

    def run(self, seq: str, reactivities: list[float]):
        # TODO: Figure out how to set things up such that we only compute bpps once for all heuristics
        bpp = bpps(seq, package=self.package_name, **self.arnie_bpp_kwargs)
        return {
            heuristic['name']: pk_predict_from_bpp(bpp, heuristic=heuristic['heuristic'], **(heuristic['arnie_heuristic_kwargs'] or {}))
            for heuristic in self.heuristics
        }

class Vienna2Predictor(ArnieMfePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('vienna_2', as_name, arnie_kwargs)
    
    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(0.014957053719891054*x + 6.521510893351619e-06*x**2),
            math.ceil(7.58615971706658e-06*x**2),
            math.ceil(4.175814836611823e-06*x**2),
            1,
            math.ceil(66661373.15594226 + 66884.62043469177*x),
            0
        )

class Contrafold2Predictor(ArnieMfePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('contrafold_2', as_name, arnie_kwargs)

    def run(self, *args, **kwargs):
        if not 'param_file' in self.arnie_kwargs:
            self.arnie_kwargs['param_file'] = os.environ['CONTRAFOLD_2_PARAMS_PATH']
        return super().run(*args, **kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(1.8780177003189198e-05*x**2),
            math.ceil(0.0017168568837895044*x + 1.370608509360984e-06*x**2 + 2.5353069010588043e-09*x**3 + 1.755250407902063e-12*x**4),
            math.ceil(3.0220495032131762e-09*x**3),
            1,
            math.ceil(47978433.97009161 + 124761.82510964583*x),
            0
        )

class EternafoldPredictor(ArnieMfePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('eternafold', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(0.9864763603353837 + 0.0008267324709392259*x + 8.336229748461046e-09*x**3),
            math.ceil(6.818589190331236e-09*x**3),
            math.ceil(3.0170732966282613e-09*x**3),
            1,
            math.ceil(47334809.015045516 + 124729.33100925815*x),
            0
        )

class RnastructurePredictor(ArnieMfePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('rnastructure', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(7.65439536672311e-05*x**2),
            math.ceil(5.060174568471981e-05*x**2),
            math.ceil(8.140817736321833e-06*x**2),
            1,
            math.ceil(164598480.16611868 + 39774.67228873097*x),
            0
        )

class E2efoldPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('e2efold', as_name, arnie_kwargs)

    # def approximate_resources(self, seq: str) -> UtilizedResources:
    #     x = len(seq)
    #     return UtilizedResources(
    #         0,
    #         0,
    #         0,
    #         1,
    #         0,
    #         0
    #     )

class HotknotsPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('hotknots', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(1.7166548614537612e-07*x**4),
            math.ceil(5.5522318537226254e-08*x**4),
            math.ceil(0.0015334729937377905*x),
            1,
            math.ceil(128977478.30623925 + 7.906412107091359e-05*x**5 + 9.210784821427999e-07*x**6),
            0
        )

class IpknotPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('ipknot', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(1.1203593366633413 + 1.506286414749126e-14*x**6),
            math.ceil(0.258856113992567 + 4.383183867322136e-15*x**6),
            math.ceil(0.00018581983943878903*x),
            1,
            math.ceil(106331571.89634839 + 24913.345732650258*x),
            0
        )

class KnottyPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('knotty', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(5.395909039368899e-07*x**4 + 1.2305318696389623e-09*x**5),
            math.ceil(4.6035433980017475e-07*x**4),
            math.ceil(1.6018657621263243e-05*x**3),
            1,
            math.ceil(102096593.10636786 + 160.85474224552664*x**3),
            0
        )

class PknotsPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('pknots', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(5.099368419606039e-10*x**6),
            math.ceil(3.4876825795207265e-08*x**5),
            math.ceil(3.046780806734162e-08*x**5),
            1,
            math.ceil(186226181.74814296 + 125212.93991127468*x + 1765.802047784336*x**2 + 20.147797467970936*x**3 + 0.20690040456199893*x**4),
            0
        )

class SpotrnaPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('spotrna', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(30.109913152533956 + 0.0010967470575417066*x**2),
            math.ceil(16.18377656041205 + 0.26514851833432623*x),
            math.ceil(25.74060961972136 + 0.024162637794247366*x + 0.0009185725454450153*x**2),
            1,
            math.ceil(897227420.570466 + 4541843.719817102*x),
            0
        )

class Spotrna2Predictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('spotrna2', as_name, arnie_kwargs)

    # def approximate_resources(self, seq: str) -> UtilizedResources:
    #     x = len(seq)
    #     return UtilizedResources(
    #         0,
    #         0,
    #         0,
    #         1,
    #         0,
    #         0
    #     )

class NupackPkPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('nupack', as_name, arnie_kwargs)
    
    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(6.356188679435512e-07*x**4),
            math.ceil(2.5782818357674076e-07*x**4),
            math.ceil(6.365939349049284e-08*x**4),
            1,
            math.ceil(4885744.735577856 + 710.1559426966696*x**3),
            0
        )

class RnastructureShapePredictor(ArnieMfeShapePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('rnastructure', as_name, arnie_kwargs)
    
    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(7.199883536370171e-05*x**2),
            math.ceil(4.468416499075908e-05*x**2),
            math.ceil(5.432031322291636e-06*x**2),
            1,
            math.ceil(155283629.4425676 + 36837.490900609955*x),
            0
        )

class ShapeknotsPredictor(ArnieMfeShapePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('rnastructure', as_name, {
            **(arnie_kwargs or {}),
            'pseudo': True
        })

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(0.0015564560910918425*x**2),
            math.ceil(0.0007664366687325299*x**2),
            math.ceil(2.6665578065750506e-05*x**2),
            1,
            math.ceil(145713272.2629301 + 162574.74522094347*x),
            0
        )

class Vienna2PkFromBppPredictor(ArniePkFromBppPredictor):
    def __init__(self, arnie_bpp_kwargs: dict = None):
        super().__init__('vienna_2', arnie_bpp_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(1.8881274197809997e-05*x**2),
            math.ceil(1.562624224698903e-05*x**2),
            math.ceil(7.511396016129651e-06*x**2),
            1,
            math.ceil(57383436.87088825 + 102710.05819533014*x),
            0
        )

class EternafoldPkFromBppPredictor(ArniePkFromBppPredictor):
    def __init__(self, arnie_bpp_kwargs: dict = None):
        super().__init__('eternafold', arnie_bpp_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(2.441166592660425e-05*x**2),
            math.ceil(1.8235190367825772e-05*x**2),
            math.ceil(7.190588857495217e-06*x**2),
            1,
            math.ceil(59833068.17404543 + 115205.90242765758*x),
            0
        )

class Contrafold2PkFromBppPredictor(ArniePkFromBppPredictor):
    def __init__(self, arnie_bpp_kwargs: dict = None):
        super().__init__('contrafold_2', arnie_bpp_kwargs)

    def run(self, *args, **kwargs):
        if not 'param_file' in self.arnie_bpp_kwargs:
            self.arnie_bpp_kwargs['param_file'] = os.environ['CONTRAFOLD_2_PARAMS_PATH']
        return super().run(*args, **kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(2.492714280992482e-05*x**2),
            math.ceil(1.8750758063515493e-05*x**2),
            math.ceil(9.628448940650172e-06*x**2),
            1,
            math.ceil(59197725.99610226 + 116193.13486723782*x),
            0
        )

# TODO: Move into Arnie
class ShapifyHfoldPredictor(Predictor):
    uses_experimental_reactivities = False

    def __init__(self, as_name: str, hfold_location: str = None):
        self.name = as_name
        self.hfold_location = hfold_location

    def run(self, seq: str, reactivities: list[float]):
        hfold_location = self.hfold_location or os.environ['HFOLD_PATH']
        hfold_return = subprocess.run([hfold_location+"/HFold_iterative", "--s", seq.strip()], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

        # Pull structure out of Shapify output
        match = re.search(r"Result_0: (.*) \n", hfold_return.stdout)
        if match:
            structure = match.group(1)
        else:
            raise Exception(f'Could not get result from hfold output (stdout: {hfold_return.stdout}, stderr: {hfold_return.stderr})')
        
        # Sanitize dbn structure to match our conventions 
        return {
            self.name: convert_bp_list_to_dotbracket(convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True), len(structure))
        }

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            math.ceil(8.349942686979703e-06*x**3),
            math.ceil(3.4616149343646593e-06*x**3),
            math.ceil(0.0002016493481831969*x**1 + 6.704175985450193e-06*x**2),
            1,
            math.ceil(105120779.00338717 + 30140.694338146823*x),
            0
        )

    @property
    def prediction_names(self):
        return [self.name]

class RibonanzaNetSSPredictor(Predictor):
    uses_experimental_reactivities=False
    gpu=True

    def __init__(self, as_name: str, env_location: str = None):
        self.name = as_name
        self.env_location = env_location

    def run(self, seq: str, reactivities: list[float]):
        env_location = self.env_location or os.environ['RIBONANZANET_ENV_PATH']
        rnet_return = subprocess.run([env_location+"/python", os.path.join(os.path.dirname(__file__), '../../../lib/inference-2d.py'), seq.strip()], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

        match = re.search(r"structure:(.*)\n", rnet_return.stdout)
        if match:
            structure = match.group(1)
        else:
            raise Exception(f'Could not get result from rnet output (stdout: {rnet_return.stdout}, stderr: {rnet_return.stderr})')
        
        return {
            self.name: structure
        }

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            9.93630267074983 + 0.009064971919693571*x,
            5.328553048504195 + 0.0034631731109770417*x,
            4.204866530717558 + 0.0022178808249596503*x,
            1,
            1332768293.9422886*x**0 + 142549.5294871579*x**1,
            1521260659.6040597 + 66443.01382998787*x + 8378.075744909687*x**2
        )

    @property
    def prediction_names(self):
        return [self.name]

class RibonanzaNetShapeDerivedPredictor(Predictor):
    uses_experimental_reactivities=False
    gpu=True

    def __init__(self, env_location: str = None):
        self.env_location = env_location
        self.configurations = []

    def add_configuration(self, reactivity_type: str, derivation_package: str, as_name: str):
        self.configurations.append({
            'reactivity_type': reactivity_type,
            'derivation_package': derivation_package,
            'name': as_name
        })
        return self

    def run(self, seq: str, reactivities: list[float]):
        env_location = self.env_location or os.environ['RIBONANZANET_ENV_PATH']
        rnet_return = subprocess.run([env_location+"/python", os.path.join(os.path.dirname(__file__), '../../../lib/inference-shape.py'), seq.strip()], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

        match = re.search(r"2a3:(.*)\ndms:(.*)\n", rnet_return.stdout)
        if match:
            reactivity_2a3 = [float(x) for x in match.group(1).split(',')]
            reactivity_dms = [float(x) for x in match.group(2).split(',')]
        else:
            raise Exception(f'Could not get result from rnet output (stdout: {rnet_return.stdout}, stderr: {rnet_return.stderr})')
        
        return {
            conf['name']: mfe(
                seq,
                'rnastructure',
                shape_signal=reactivity_2a3 if conf['reactivity_type'] == '2a3' else reactivity_dms,
                pseudo=conf['derivation_package'] == 'shapeknots'
            )
            for conf in self.configurations
        }

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            0.0017890144980402386*x**2,
            0.0010536675291728619*x**2,
            2.686718054064782 + 0.007261345887062982*x + 3.156924098456259e-05*x**2,
            1,
            # Note that the sampling didn't include this second term - I lifted it from shapeknots.
            # presumably the allocations from rnet itself wound up dwarfing it? I've included the length
            # dependent term myself as a safety measure
            1064575385.5 + 162574.74522094347*x,
            1517377492.8717134 + 8709.613988478806*x**2
        )

    @property
    def prediction_names(self):
        return [conf['name'] for conf in self.configurations]
