# Release announcements

Each release has a short post here, `docs/releases/<version>.md`: the text to paste into Discord and other places. GitHub Releases shows it first, above the release's full notes.

Copy the shape of the newest file:

```md
tensorless v<version> is out!
tensorless: tuned models for the Tesla P100, in one command.

https://github.com/iammrduncan/tensorless/releases/tag/v<version>

Before upgrading: <what a user must know or do first; leave the line out when there is nothing>

Changes:

- New model: <Name> => <what it gives you>
- <one plain line for each thing a user would notice>
- Bug fix: <what stopped going wrong>
```

- The first line is exactly `tensorless v<version> is out!`.
- The second line is the home page's title, so a post shared on its own makes sense to someone who has never heard of tensorless. When the title changes, so does this line.
- **Before upgrading** goes above **Changes**, and only when there is something to know or do.
- One line for each thing a user would notice, the biggest first. Numbers only as the receipts give them.
- Plain Markdown that reads the same on GitHub and in Discord: no tables, no nested lists, no bold.
- Write it before the tag: the release workflow takes this file as the release's notes when it exists. The full notes (launch targets, what's new, install, known limits) go below it on the release page, after a `---` line.
