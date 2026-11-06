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
            '';
          };
        });

      checks = forAllSystems (system: {
        gputaskman = self.packages.${system}.gputaskman;
      });
    };
}
