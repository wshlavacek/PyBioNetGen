from bionetgen.core.exc import BNGParseError


class RuleMod:
    """
    Rule modifiers class for storage and printing.
    """

    def __init__(self, mod_type=None, modifiers=None) -> None:
        # valid mod types
        self.valid_mod_names = ["DeleteMolecules", "MoveConnected", "TotalRate"]
        self.modifiers: list[str] = []
        self.type = mod_type
        if modifiers is not None:
            for modifier in modifiers:
                self.add_modifier(modifier)

    def __str__(self) -> str:
        if len(self.modifiers) > 0:
            return " ".join(self.modifiers)
        if self.type is None:
            return ""
        return self.type  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"Rule modifier of type {self.type}"

    def add_modifier(self, modifier) -> None:
        text = str(modifier).strip()
        if text and text not in self.modifiers:
            self.modifiers.append(text)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, val):
        if val in self.valid_mod_names or val is None:
            self._type = val
        else:
            raise BNGParseError(message=f": Rule modifier type {val} is not a valid type")
