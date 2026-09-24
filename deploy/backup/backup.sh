#!/bin/sh
# kalos SQLite backup sidecar loop. Runs inside the `backup` service in
# compose.yaml, sharing the engine's state volume (read-only) plus a
# host-mounted backup directory.
#
# WHY .backup AND NOT `cp`/`cat`/`tar` OF THE LIVE .db FILES
# ------------------------------------------------------------------------
# kalos's two SQLite stores (kalos/store/sqlite_store.py's experiments.db,
# kalos/portal/campaign.py's portal.db) do not set `journal_mode=WAL` -
# both open plain `sqlite3.connect(...)` connections, which leaves SQLite's
# default rollback-journal mode in effect. Under that mode a write in
# progress can, at any instant, have the main .db file and a same-named
# `-journal` sidecar file in an interdependent, transiently-inconsistent
# pairing; a file-level copy of just the .db file can land mid-write and
# either capture a torn page or miss the journal entirely, producing a
# backup that LOOKS like a valid SQLite file (it opens) but is silently
# missing or corrupting the transaction that was in flight - the worst kind
# of backup failure because nothing about it fails loudly.
#
# `sqlite3 <db> ".backup <dest>"` instead uses SQLite's own Online Backup
# API: it takes SQLite's own internal read lock and copies committed pages
# through SQLite itself, producing a transactionally-consistent snapshot
# even while the source is being written concurrently by the engine
# process. This is the correct tool specifically because kalos runs in
# rollback-journal mode, not WAL - see kalos/store/sqlite_store.py and
# kalos/portal/campaign.py for where that mode is (implicitly) chosen.
#
# The per-tenant "latest analysis" JSON cache under `<state>/latest/*.json`
# is explicitly best-effort persistence in the source itself
# (kalos/portal/app.py's `_save_latest`: "Disk persistence is best-effort;
# in-memory still serves this run") and is fully reconstructible by
# re-running the last analysis, so it does not get the same backup rigor -
# it is copied opportunistically below, not treated as a restore-critical
# artifact.
set -eu

# Default matches deploy/engine/Containerfile's KALOS_STATE_DIR (/home/kalos/.kalos) -
# see that file's comment for the current state: SqliteStore
# (experiments.db), the runner lock, CampaignStore (portal.db), and the
# `_LATEST` cache all now read KALOS_STATE_DIR at call time, so this only
# needs to track whatever the engine container is actually configured with,
# not hardcode a path those modules ignore. The literal default here still
# matches the engine's, so the two stay in step even if someone runs this
# script with no KALOS_STATE_DIR set at all.
STATE_DIR="${KALOS_STATE_DIR:-/home/kalos/.kalos}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
KEEP_DAYS="${BACKUP_KEEP_DAYS:-7}"
INTERVAL="${BACKUP_INTERVAL_SECONDS:-86400}"

log() {
    printf '%s backup: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

backup_db() {
    name="$1"
    src="$STATE_DIR/$name.db"
    if [ ! -f "$src" ]; then
        log "$src not present yet (nothing uploaded/run since last start) - skipping $name"
        return 0
    fi
    dest_dir="$BACKUP_DIR/$name"
    mkdir -p "$dest_dir"
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    dest="$dest_dir/${name}-${ts}.db"
    tmp="$dest.tmp"
    # Online Backup API - safe against a concurrently-writing engine process,
    # unlike a plain file copy. See header comment for why.
    if sqlite3 "$src" ".backup '$tmp'"; then
        mv "$tmp" "$dest"
        log "wrote $dest ($(du -h "$dest" | cut -f1))"
    else
        rm -f "$tmp"
        log "FAILED to back up $src - left previous backups untouched"
        return 1
    fi
}

backup_latest_cache() {
    src="$STATE_DIR/latest"
    [ -d "$src" ] || return 0
    dest_dir="$BACKUP_DIR/latest-cache"
    mkdir -p "$dest_dir"
    # Best-effort, per the header comment - a plain recursive copy is fine
    # here because this cache is not the durable record of anything; the
    # sqlite stores are.
    cp -r "$src" "$dest_dir/$(date -u +%Y%m%dT%H%M%SZ)" 2>/dev/null || true
}

prune_old() {
    # Applies uniformly under $BACKUP_DIR: any backup artifact (sqlite
    # snapshot or latest-cache directory) older than KEEP_DAYS is removed.
    find "$BACKUP_DIR" -mindepth 2 -maxdepth 2 -mtime "+$KEEP_DAYS" -exec rm -rf {} + 2>/dev/null || true
}

log "starting - state=$STATE_DIR backups=$BACKUP_DIR keep_days=$KEEP_DAYS interval=${INTERVAL}s"

while true; do
    backup_db experiments
    backup_db portal
    backup_latest_cache
    prune_old
    sleep "$INTERVAL"
done
