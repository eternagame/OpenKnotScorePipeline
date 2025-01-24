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
        processed_reactivities = [value if value is not None else 0 for value in reactivities]
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

    def add_heuristic(self, heuristic, as_name: str, arnie_heuristic_kwargs: dict = None):
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
            9.363249628553956e-06*x**2,
            6.435575569102438e-06*x**2,
            1.999630031075256e-07*x**2 + 1.291168337955278e-09*x**3,
            1,
            74529341.14143184 + 50039.5689844508*x,
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
            1.3555230088408226e-05*x**2,
            0.0017168568837895044*x + 1.370608509360984e-06*x**2 + 2.5353069010588043e-09*x**3 + 1.755250407902063e-12*x**4,
            3.0220495032131762e-09*x**3,
            1,
            67375207.71532412*x + 85637.61691465044*x,
            0
        )

class EternafoldPredictor(ArnieMfePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('eternafold', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            1.3091725201772042e-05*x**2,
            0.0016406025265967043*x + 1.3297639187003973e-06*x**2 + 2.614209212056624e-09*x**3 + 1.6809009331685289e-12*x**4,
            3.0170732966282613e-09*x**3,
            1,
            67076985.533487394 + 85679.15721484796*x,
            0
        )

class RnastructurePredictor(ArnieMfePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('rnastructure', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            6.115858668137446e-05*x**2,
            3.831539039645807e-05*x**2,
            5.4581028524413525e-06*x**2,
            1,
            148235610.48851812 + 27.12283679855023*x**2,
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
            1.034993732123478e-05*x**2 + 9.25764975270185e-06*x**3,
            3.854509815534424e-06*x**3,
            0.0015334729937377905*x,
            1,
            98192274.35077795 + 155706.9184299503*x,
            0
        )

class IpknotPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('ipknot', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            1.1203593366633413 + 1.506286414749126e-14*x**6,
            0.258856113992567 + 4.383183867322136e-15*x**6,
            0.00018581983943878903*x,
            1,
            106331571.89634839 + 24913.345732650258*x,
            0
        )

class KnottyPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('knotty', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            8.749532960152117e-07*x**4,
            3.8617693999595035e-05*x**3,
            1.0145749847475423e-05*x**3,
            1,
            102096593.10636786 + 160.85474224552664*x**3,
            0
        )

class PknotsPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('pknots', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            0.48383213130062275 + 6.676609386480957e-12*x**7,
            0.006871826653624754*x + 3.490906504167486e-10*x**6,
            0.007986320659648508*x + 2.9283003948705727e-10*x**6,
            1,
            172387174.34916383 + 70428.26919995953*x + 1273.275311583844*x**2 + 18.769368117673764*x**3 + 0.24193155373736816*x**4,
            0
        )

class SpotrnaPredictor(ArniePkPredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('spotrna', as_name, arnie_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            30.109913152533956 + 0.0010967470575417066*x**2,
            16.18377656041205 + 0.26514851833432623*x,
            25.74060961972136 + 0.024162637794247366*x + 0.0009185725454450153*x**2,
            1,
            1038528677.5501341 + 3053988.952766466*x,
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
            5.065959329800869e-05*x**3,
            1.798339809862075e-05*x**3,
            5.6110885356256905e-06*x**3,
            1,
            26467653.162968375 + 616.2855681555868*x**3,
            0
        )

class RnastructureShapePredictor(ArnieMfeShapePredictor):
    def __init__(self, as_name: str, arnie_kwargs: dict = None):
        super().__init__('rnastructure', as_name, arnie_kwargs)
    
    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            6.21061110521885e-05*x**2,
            3.8566207125151296e-05*x**2,
            5.432031322291636e-06*x**2,
            1,
            155283629.4425676 + 36837.490900609955*x,
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
            0.0012417930499900851*x**2,
            0.0006240957445198866*x**2,
            2.4251054138648075e-05*x**2,
            1,
            145713272.2629301 + 162574.74522094347*x,
            0
        )

class Vienna2PkFromBppPredictor(ArniePkFromBppPredictor):
    def __init__(self, arnie_bpp_kwargs: dict = None):
        super().__init__('vienna_2', arnie_bpp_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            1.7113306135215377e-05*x**2,
            1.3683964534960524e-05*x**2,
            6.232218465095017e-06*x**2,
            1,
            67578509.42869748 + 81169.52123215972*x,
            0
        )

class EternafoldPkFromBppPredictor(ArniePkFromBppPredictor):
    def __init__(self, arnie_bpp_kwargs: dict = None):
        super().__init__('eternafold', arnie_bpp_kwargs)

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            2.051440636442e-05*x**2,
            1.4867130298397484e-05*x**2,
            7.190588857495217e-06*x**2,
            1,
            71155137.5299398 + 85096.09312683524*x,
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
            2.0519892402473385e-05*x**2,
            1.4987091803218787e-05*x**2,
            7.2597793931347964e-06*x**2,
            1,
            71373510.4481341 + 84211.14104063316*x,
            0
        )

# TODO: Move into Arnie
class ShapifyHfoldPredictor(Predictor):
    uses_experimental_reactivities = False
    name = ''

    def __init__(self, as_name: str, hfold_location: str = None):
        self.name = as_name
        self.hfold_location = hfold_location

    def run(self, seq: str, reactivities: list[float]):
        hfold_location = self.hfold_location or os.environ['HFOLD_PATH']
        hfold_return = subprocess.run([hfold_location+"/HFold_iterative", "--s", seq.strip()], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

        # Pull structure out of Shapify output
        match = re.search(r"Result_0: (.*) \n", hfold_return.stdout)
        structure = match.group(1) if match else "ERR"
        
        # Sanitize dbn structure to match our conventions 
        return {
            self.name: convert_bp_list_to_dotbracket(convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True), len(structure))
        }

    def approximate_resources(self, seq: str) -> UtilizedResources:
        x = len(seq)
        return UtilizedResources(
            8.349942686979703e-06*x**3,
            3.4616149343646593e-06*x**3,
            0.0002016493481831969*x**1 + 6.704175985450193e-06*x**2,
            1,
            105120779.00338717 + 30140.694338146823*x,
            0
        )

    @property
    def prediction_names(self):
        return [self.name]
