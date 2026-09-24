# syntax=docker/dockerfile:1
#
# kalos backup sidecar - alpine + the sqlite3 CLI (for the Online Backup API,
# `.backup`) is all this needs; no Python, no kalos package import. See
# backup.sh for why `.backup` and not a plain file copy, and ../RUNBOOK.md
# for the restore drill.
FROM alpine:3.20

RUN apk add --no-cache sqlite coreutils \
    && addgroup -S kalos && adduser -S -G kalos -u 10001 kalos

COPY backup.sh /usr/local/bin/backup.sh
RUN chmod +x /usr/local/bin/backup.sh

USER kalos

ENTRYPOINT ["/usr/local/bin/backup.sh"]
