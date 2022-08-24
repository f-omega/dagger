self: super:
let overlayPython = python: python.override {
      packageOverrides = self: super: {
        pytypes = self.callPackage ./pytypes.nix {};
        pylsp-rope = self.callPackage ./pylsp-rope.nix {};
      };
    };
in {
  python39 = overlayPython super.python39;
  python310 = overlayPython super.python310;
}
