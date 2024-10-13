# Command line interface
yarPPG comes with the `run-yarppg` command to launch a graphical user
interface. The command line interface is built as a
*structured configuration* using [Hydra](https://hydra.cc).

Hydra offers a robust configuration management with type checking for
complex, modular settings hierarchies. Additionally, we get a
powerful override syntax, allowing users to adjust the settings
from the command line.

!!! note

    The command line interface is still a work in progress. For now, you
    can adjust only a handful of options.

You can call `run-yarppg --help` to get more information of available
options and how to override them:

```
run-yarppg is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

ui: qt6_simple, simplest


== Config ==
Override anything in the config (foo.bar=value)

ui:
  roi_alpha: 0.0
  video: 0
savepath: null
detector: facemesh
filter:
  fs: 30.0
  f1: 0.5
  f2: 2.0
  btype: bandpass
  ftype: butter
  order: 2
algorithm: green


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```
