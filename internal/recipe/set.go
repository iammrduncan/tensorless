package recipe

import (
	"fmt"
	"io/fs"
	"path"
	"regexp"
	"sort"
	"strings"
)

// Set is every recipe the launcher knows, sorted by repo id, then card.
type Set struct {
	Recipes []*Recipe
}

// Load reads <org>/<repo>/<card>.toml from fsys, whose root is a recipes
// directory, and ignores every other file. label prefixes each recipe's
// Source. Any invalid recipe fails the whole load.
func Load(fsys fs.FS, label string) (*Set, error) {
	s := &Set{}
	orgs, err := readDirs(fsys, ".")
	if err != nil {
		return nil, fmt.Errorf("reading recipes in %s: %w", label, err)
	}
	for _, org := range orgs {
		repos, err := readDirs(fsys, org)
		if err != nil {
			return nil, fmt.Errorf("reading recipes in %s: %w", label, err)
		}
		for _, repo := range repos {
			dir := path.Join(org, repo)
			files, err := fs.ReadDir(fsys, dir)
			if err != nil {
				return nil, fmt.Errorf("reading recipes in %s: %w", label, err)
			}
			for _, f := range files {
				if f.IsDir() || !strings.HasSuffix(f.Name(), ".toml") || strings.HasPrefix(f.Name(), ".") {
					continue
				}
				rel := path.Join(dir, f.Name())
				src := path.Join(label, rel)
				data, err := fs.ReadFile(fsys, rel)
				if err != nil {
					return nil, fmt.Errorf("reading %s: %w", src, err)
				}
				r, err := Parse(data, src)
				if err != nil {
					return nil, err
				}
				if r.Model.HFRepo != dir {
					return nil, fmt.Errorf("%s: model.hf_repo %q must match its directory %q", src, r.Model.HFRepo, dir)
				}
				if card := strings.TrimSuffix(f.Name(), ".toml"); r.Card.ID != card {
					return nil, fmt.Errorf("%s: card.id %q must match its file name %q", src, r.Card.ID, card)
				}
				s.Recipes = append(s.Recipes, r)
			}
		}
	}
	if len(s.Recipes) == 0 {
		return nil, fmt.Errorf("no recipes found in %s (expected <org>/<repo>/<card>.toml files)", label)
	}
	sort.Slice(s.Recipes, func(i, j int) bool {
		a, b := strings.ToLower(s.Recipes[i].ID()), strings.ToLower(s.Recipes[j].ID())
		return a < b
	})
	if err := s.checkNames(); err != nil {
		return nil, err
	}
	return s, nil
}

func readDirs(fsys fs.FS, dir string) ([]string, error) {
	entries, err := fs.ReadDir(fsys, dir)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			out = append(out, e.Name())
		}
	}
	return out, nil
}

// checkNames refuses repo ids that differ only in case, and a short name
// used by two repos: either would make a MODEL argument ambiguous.
func (s *Set) checkNames() error {
	repos := map[string]string{}
	names := map[string]string{}
	for _, r := range s.Recipes {
		key := strings.ToLower(r.Repo())
		if other, ok := repos[key]; ok && other != r.Repo() {
			return fmt.Errorf("%s: repo ids %q and %q differ only in case", r.Source, other, r.Repo())
		}
		repos[key] = r.Repo()
		if r.Model.Name == "" {
			continue
		}
		if _, ok := repos[r.Model.Name]; ok {
			return fmt.Errorf("%s: model.name %q is also a repo id", r.Source, r.Model.Name)
		}
		if other, ok := names[r.Model.Name]; ok && !strings.EqualFold(other, r.Repo()) {
			return fmt.Errorf("%s: model.name %q is already the short name of %s", r.Source, r.Model.Name, other)
		}
		names[r.Model.Name] = r.Repo()
	}
	return nil
}

// Models returns the repo ids, sorted, without repeats.
func (s *Set) Models() []string {
	var out []string
	for _, r := range s.Recipes {
		if len(out) == 0 || out[len(out)-1] != r.Repo() {
			out = append(out, r.Repo())
		}
	}
	return out
}

// Cards returns the card ids, sorted, without repeats.
func (s *Set) Cards() []string {
	seen := map[string]bool{}
	var out []string
	for _, r := range s.Recipes {
		if !seen[r.Card.ID] {
			seen[r.Card.ID] = true
			out = append(out, r.Card.ID)
		}
	}
	sort.Strings(out)
	return out
}

// Matches reports whether a MODEL argument (without @REVISION) names r: its
// repo id, compared without case as the Hub does, or its short name.
func (r *Recipe) Matches(model string) bool {
	return strings.EqualFold(model, r.Model.HFRepo) || (r.Model.Name != "" && strings.EqualFold(model, r.Model.Name))
}

// ForModel returns the recipes for a MODEL argument, sorted by card. If
// nothing matches, the error lists the known repo ids.
func (s *Set) ForModel(model string) ([]*Recipe, error) {
	var out []*Recipe
	for _, r := range s.Recipes {
		if r.Matches(model) {
			out = append(out, r)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no recipe for %q. Known models: %s. See: tensorless models", model, strings.Join(s.Models(), ", "))
	}
	return out, nil
}

// Lookup returns the recipe for model on card. If there is none, the error
// lists what exists.
func (s *Set) Lookup(model, card string) (*Recipe, error) {
	rs, err := s.ForModel(model)
	if err != nil {
		return nil, err
	}
	var cards []string
	for _, r := range rs {
		if r.Card.ID == card {
			return r, nil
		}
		cards = append(cards, r.Card.ID)
	}
	return nil, fmt.Errorf("no recipe for %s on card %q. %s has recipes for: %s. Known cards: %s",
		rs[0].Repo(), card, rs[0].Repo(), strings.Join(cards, ", "), strings.Join(s.Cards(), ", "))
}

var reCommit = regexp.MustCompile(`^[0-9a-f]{40}$`)

// SplitModel splits a MODEL argument, ORG/REPO[@REVISION] or a short name,
// into the name and the revision ("" when none is given). A revision must be
// a 40-character commit.
func SplitModel(arg string) (model, revision string, err error) {
	model, revision, found := strings.Cut(arg, "@")
	if model == "" {
		return "", "", fmt.Errorf("MODEL must be a repo id such as iamMrDuncan/Qwen3.8-27B-PN3-GGUF, not %q", arg)
	}
	if !found {
		return model, "", nil
	}
	revision = strings.ToLower(revision)
	if !reCommit.MatchString(revision) {
		return "", "", fmt.Errorf("the revision in %q must be a 40-character commit, not a branch or tag", arg)
	}
	return model, revision, nil
}
