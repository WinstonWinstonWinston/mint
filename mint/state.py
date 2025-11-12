from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from mint.data.dataset import MINTDataset
from mint.module import MINTModule
from mint.utils import set_seed
import random, uuid

@dataclass(slots=True)
class MINTState:
    r"""
    Minimal experiment state holding a model, datasets, RNG, and extensible attachments.

    :param seed:
        Seed to initialize the pseudo-random number generator.
    :type seed: Optional[int]

    :param run_id:
        Unique identifier for this state instance. Set during initialization.
    :type run_id: str

    :param rng:
        Pseudo-random number generator initialized from :py:data:`seed`.
    :type rng: random.Random

    :param module:
        Training or inference module.
    :type module: Optional[MINTModule]

    :param dataset_train:
        Training dataset.
    :type dataset_train: Optional[MINTDataset]

    :param dataset_valid:
        Validation dataset.
    :type dataset_valid: Optional[MINTDataset]

    :param dataset_test:
        Test dataset.
    :type dataset_test: Optional[MINTDataset]

    :param extras:
        Open-ended storage for named objects such as experiments, priors, samples, etc.
    :type extras: Dict[str, Any]

    :param strict_extras:
        If ``True``, :py:meth:`set` only updates existing keys in :py:data:`extras`.
    :type strict_extras: bool
    """
    # core
    seed: Optional[int] = None
    run_id: str = field(init=False)
    rng: random.Random = field(init=False, repr=False)

    # primary artifacts
    module: Optional[MINTModule] = None
    dataset_train: Optional[MINTDataset] = None
    dataset_valid: Optional[MINTDataset] = None
    dataset_test: Optional[MINTDataset] = None

    # arbitrary attachments (experiments, priors, samples, etc.)
    extras: Dict[str, Any] = field(default_factory=dict, repr=False)

    # optional guard to forbid creating new keys via set()
    strict_extras: bool = False

    def __post_init__(self) -> None:
        r"""
        Finalize initialization by assigning a unique :py:data:`run_id` and seeding :py:data:`rng`.

        :return:
            ``None``.
        :rtype: None
        """
        self.run_id = str(uuid.uuid4())
        s = self.seed if self.seed is not None else random.randrange(1 << 30)
        set_seed(s)

    # lookups
    def get(self, key: str, default: Any = None) -> Any:
        r"""
        Retrieve an item from :py:data:`extras`.

        :param key:
            Name of the item to fetch.
        :type key: str

        :param default:
            Value to return if ``key`` is absent.
        :type default: Any

        :return:
            Stored value if present, otherwise ``default``.
        :rtype: Any
        """
        return self.extras.get(key, default)

    def require(self, key: str) -> Any:
        r"""
        Retrieve a required item from :py:data:`extras`.

        :param key:
            Name of the item to fetch.
        :type key: str

        :return:
            Stored value for ``key``.
        :rtype: Any

        :raises KeyError:
            If ``key`` is not present in :py:data:`extras`.
        """
        if key not in self.extras:
            raise KeyError(f"{key} not in state.extras")
        return self.extras[key]

    # mutation
    def set(self, key: str, value: Any) -> None:
        r"""
        Insert or update a single item in :py:data:`extras`.

        :param key:
            Name of the item to set.
        :type key: str

        :param value:
            Value to associate with ``key``.
        :type value: Any

        :return:
            ``None``.
        :rtype: None

        :raises KeyError:
            If :py:data:`strict_extras` is ``True`` and ``key`` does not already exist.
        """
        if self.strict_extras and key not in self.extras:
            raise KeyError(f"{key} not in state.extras (strict mode)")
        self.extras[key] = value

    def attach(self, **named_objects: Any) -> MINTState:
        r"""
        Bulk insert one or more named objects into :py:data:`extras`.

        :param named_objects:
            Keyword arguments mapping names to objects to store.
        :type named_objects: Any

        :return:
            The current state for chaining.
        :rtype: MintState
        """
        self.extras.update(named_objects)
        return self

    # convenience
    def has(self, key: str) -> bool:
        r"""
        Check if a key exists in :py:data:`extras`.

        :param key:
            Name to check.
        :type key: str

        :return:
            ``True`` if the key exists, else ``False``.
        :rtype: bool
        """
        return key in self.extras

    def pop(self, key: str, default: Any = None) -> Any:
        r"""
        Remove and return an item from :py:data:`extras`.

        :param key:
            Name of the item to remove.
        :type key: str

        :param default:
            Value to return if ``key`` is absent.
        :type default: Any

        :return:
            The removed value if present, otherwise ``default``.
        :rtype: Any
        """
        return self.extras.pop(key, default)
