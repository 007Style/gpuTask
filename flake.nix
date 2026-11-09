{
  description = "GPU Task Manager for Linux";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      packages = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          project = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project;
          pythonSource = path: pkgs.lib.fileset.fileFilter
            (file: !(pkgs.lib.hasSuffix ".pyc" file.name || pkgs.lib.hasSuffix ".pyo" file.name)) path;
        in {
          default = self.packages.${system}.gputaskman;
          gputaskman = pkgs.python3Packages.buildPythonApplication {
            pname = project.name;
            version = project.version;
            pyproject = true;
            src = pkgs.lib.fileset.toSource {
              root = ./.;
              fileset = pkgs.lib.fileset.unions [
                ./pyproject.toml ./README.md ./LICENSE
                (pythonSource ./src) (pythonSource ./tests) (pythonSource ./tools)
              ];
            };
            nativeBuildInputs = [ pkgs.makeWrapper ];
            build-system = [ pkgs.python3Packages.uv-build ];
            # Nixpkgs supplies the backend; accept its newer release only in
            # this build. The resulting wheel and the full test suite are checked.
            postPatch = ''
              substituteInPlace pyproject.toml \
                --replace-fail 'uv_build>=0.9.26,<0.10.0' 'uv_build>=0.9.26,<0.13.0'
            '';
            dependencies = with pkgs.python3Packages; [ nvidia-ml-py psutil textual ];
            checkPhase = ''
              runHook preCheck
              python -m unittest discover -s tests -q
              runHook postCheck
            '';
            pythonImportsCheck = [ "gputaskman.cli" "gputaskman.app" ];
            # WSL2 exposes the host driver's NVML in /usr/lib/wsl/lib (not in
            # /run/opengl-driver/lib or the loader's default path), so a
            # Nix-built Python cannot find libnvidia-ml.so.1 and reports no
            # GPU. Put that directory on the wrapper's loader path; the path
            # does not exist on NixOS or native Linux, where it is ignored.
            postFixup = ''
              wrapProgram $out/bin/gputaskman --suffix LD_LIBRARY_PATH : /usr/lib/wsl/lib
              wrapProgram $out/bin/gtm --suffix LD_LIBRARY_PATH : /usr/lib/wsl/lib
            '';
            meta = {
              inherit (project) description;
              homepage = project.urls.Repository;
              license = pkgs.lib.licenses.mit;
              mainProgram = "gputaskman";
              platforms = systems;
            };
          };
        });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.gputaskman}/bin/gputaskman";
          meta.description = "GPU Task Manager";
        };
      });

      devShells = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in {
          default = pkgs.mkShell {
            packages = [
              (pkgs.python3.withPackages (ps: with ps; [ nvidia-ml-py psutil textual uv-build ]))
              pkgs.uv
            ];
            shellHook = ''
              export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
              # WSL2 keeps NVML outside the loader's default path; see postFixup above.
              if [ -d /usr/lib/wsl/lib ]; then
                export LD_LIBRARY_PATH="/usr/lib/wsl/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              fi
            '';
          };
        });

      checks = forAllSystems (system: {
        gputaskman = self.packages.${system}.gputaskman;
      });
    };
}
