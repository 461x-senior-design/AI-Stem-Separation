# scripts/configs

Named env configs consumed by the `stemmy` CLI.

## Layout

```
scripts/configs/
  shared/              committed — everyone sees these
    recon-p10.env
    kl-p10.env
    ...
  <user>/              per-user, git-ignored by default
    defaults.env       your personal overrides on top of scripts/defaults.env
    <name>.env         your named experiments
```

`<user>` is resolved from `$STEMMY_USER`, then `$USER`, then `$ONID`.

## Resolution order 

1. `scripts/defaults.env` — project baseline (committed)
2. `scripts/configs/<user>/defaults.env` — your personal baseline
3. `scripts/configs/<user>/<name>.env` **or** `scripts/configs/shared/<name>.env`
   (the user dir wins on collision)
4. CLI `KEY=VALUE` overrides passed to `stemmy run`

## Managing configs

```
stemmy config new my-tweak --from recon-p10   # copy recon-p10 into your user dir
stemmy config ls                              # list visible configs with shadow markers
stemmy config show my-tweak                   # merged env with per-line provenance
stemmy config edit my-tweak                   # open in $EDITOR
```

`--shared` promotes a new/edited config to `shared/` (committed). Without it,
the file lands in your user dir.

## Secrets

**Do not put `WANDB_API_KEY` or any other secret in a shared config.** Put it
in `scripts/configs/<user>/defaults.env` (ignored by git) or export it from
your shell. The CLI treats an unset `WANDB_API_KEY` as "run in offline mode"
rather than failing, so the shared configs stay clean.
