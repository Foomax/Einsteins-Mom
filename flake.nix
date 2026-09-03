{
  description = "Einstein's Mom — dev shell for PyTorch experiments on the RTX 3090";

  # Same nixpkgs *branch* as the system config (~/Projects/nixos), so the
  # toolchain stays close to what everything else here was built against. Note
  # this has its own flake.lock and so pins its own revision — it does not track
  # the system's lock. Run `nix flake update` deliberately, not incidentally.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      # Libraries the upstream PyTorch wheels expect to find the FHS way. The
      # wheels bundle their own CUDA runtime (the nvidia-* pip packages), so
      # CUDA itself is deliberately NOT listed here — adding nixpkgs' CUDA would
      # risk two runtimes fighting over the same symbols.
      wheelLibs = with pkgs; [
        stdenv.cc.cc.lib # libstdc++, the one every wheel needs
        zlib
        glib
        libGL # pulled in by opencv-style transitive deps
        libx11 # note: `xorg.libX11` is deprecated in this nixpkgs
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312 # 3.12, not 3.13: better ML wheel coverage
          uv
          git
        ];

        # libcuda.so.1 is the one library the wheels cannot bundle, because it
        # has to match the running kernel module. On NixOS the active driver
        # publishes it here. This path is a symlink that only resolves once a
        # generation with the NVIDIA driver is booted — if torch reports no GPU,
        # check that this directory actually contains libcuda.so.1.
        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath wheelLibs}:/run/opengl-driver/lib";

        # Keeps uv from trying to manage its own interpreter downloads, which
        # would land a non-NixOS-linked python in the venv.
        UV_PYTHON_DOWNLOADS = "never";
        UV_PYTHON = "${pkgs.python312}/bin/python3.12";

        shellHook = ''
          echo "einsteins-mom dev shell"
          echo "  python: $(python3 --version)"
          if [ -e /run/opengl-driver/lib/libcuda.so.1 ]; then
            echo "  libcuda: found"
          else
            echo "  libcuda: NOT FOUND -- reboot into the NVIDIA generation (see handoff.md §0)"
          fi
          [ -d .venv ] && echo "  venv: .venv (run 'source .venv/bin/activate')"
        '';
      };
    };
}
