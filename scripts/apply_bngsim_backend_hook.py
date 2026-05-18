"""Vendor the BNGsim backend hooks into the bundled BNG2.pl distributions.

PyBioNetGen's BNGsim backend-hook route keeps BNG2.pl as the BNGL action
driver and advertises a JSON helper through the environment. For that route
to reach BNGsim, the bundled BNG2.pl must be hook-aware: its simulation
actions delegate the atomic simulation job to the helper instead of running
run_network / NFsim.

There are two hook sites in ``Perl2/BNGAction.pm``:

  * **network (ode/ssa/psa)** — in ``sub simulate``, before the "Determine
    index of last rule iteration" block. Delegates a ``.net`` artifact job.
  * **network-free (nf/rm)** — in ``sub simulate_nf``, after the NFsim
    ``-sim``/``-oSteps`` arguments are built. Delegates a ``bng-xml``
    artifact job, then returns before BNG2.pl's ``.species`` read-back (the
    BNGsim backend does not yet emit a ``.species`` file — see RuleMonkey#9
    / PyBNF-Private#38), warning if get_final_state was requested.

Both hooks are inert unless ``BIONETGEN_BNGSIM_BACKEND`` is set, so a normal
BNG2.pl subprocess run is unaffected. ``pla`` has no BNGsim substitute and is
never delegated.

Each hook is wrapped in ``# >>> ... >>>`` / ``# <<< ... <<<`` bracket
markers so this script is idempotent: re-running it strips the old blocks
and re-inserts the current ones. Re-run after sync_bng_perl_from_source.py
refreshes the vendored Perl tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BNG_DISTS = ("bng-mac", "bng-linux", "bng-win")

# Legacy (pre-bracket) network hook marker, for one-time migration.
_LEGACY_MARKER = "    # PyBioNetGen/BNGsim backend hook."
_NETWORK_NEEDLE = "    # Determine index of last rule iteration\n"

# ── Network hook (ode/ssa/psa) — in sub simulate ───────────────────
_NETWORK_BODY = r'''    # BNG2.pl has already normalized model state, artifact, method, options.
    if ($ENV{'BIONETGEN_BNGSIM_BACKEND'} && $method =~ /^(cvode|ssa|psa)$/)
    {
        my @helper_command;
        if ($ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER'})
        {
            @helper_command = ($ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER'});
        }
        elsif ($ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER_PYTHON'} && $ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER_MODULE'})
        {
            @helper_command = (
                $ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER_PYTHON'},
                '-m',
                $ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER_MODULE'},
            );
        }
        else
        {
            return "BIONETGEN_BNGSIM_BACKEND_HELPER is not set.";
        }

        my $backend_method = ($method eq 'cvode') ? 'ode' : $method;
        my %sim_options = (
            t_start => $t_start,
            t_end => $t_end,
            n_steps => $n_steps,
            seed => $seed,
            print_CDAT => $print_cdat,
            print_functions => $print_fdat,
            'continue' => ($continue ? 1 : 0),
        );
        if ($method eq 'cvode')
        {
            $sim_options{atol} = $atol;
            $sim_options{rtol} = $rtol;
            $sim_options{sparse} = $sparse;
            $sim_options{steady_state} = $steady_state;
        }
        if ($method eq 'psa')
        {
            $sim_options{poplevel} = $params->{poplevel};
        }
        # An explicit sample_times array (BNG2.pl has already sorted it
        # and, if t_end was given, appended t_end). When present, n_steps
        # is undefined and these are the output times.
        if (@sample_times)
        {
            $sim_options{sample_times} = \@sample_times;
        }

        my %job = (
            artifact_path => File::Spec->rel2abs($netfile),
            artifact_format => 'net',
            method => $backend_method,
            simulation_options => \%sim_options,
            output_prefix => File::Spec->rel2abs($prefix),
            backend_flags => {
                interpreted_by => 'BNG2.pl',
                command => \@command,
            },
        );
        my $job_file = "$prefix.bngsim-job.json";
        open(my $job_fh, '>', $job_file)
            or return "Could not open BNGsim backend job file $job_file: $!";
        print $job_fh encode_json(\%job);
        close($job_fh);

        print "Running BNGsim backend helper: @helper_command $job_file\n";
        my $rc = system(@helper_command, $job_file);
        if ($rc != 0)
        {
            return sprintf("BNGsim backend helper failed with status %s.", $rc);
        }

        if ( $model->RxnList and -e "$prefix.cdat" )
        {
            print "Updating species concentrations from $prefix.cdat\n";
            open CDAT, '<', "$prefix.cdat";
            my $last_line = '';
            while (my $line = <CDAT>) {  $last_line = $line;  }
            close CDAT;
            my $conc;
            ($t_end, @$conc) = split ' ', $last_line;
            my $species = $model->SpeciesList->Array;
            unless ( $#$conc == $#$species )
            {
                return sprintf "Number of species in model (%d) and CDAT file (%d) differ", scalar @$species, scalar @$conc;
            }
            $model->Concentrations( $conc );
            $model->UpdateNet(1);
        }
        elsif ( $model->RxnList )
        {
            return "CDAT file is missing";
        }
        $model->Time($t_end);
        return '';
    }
'''

# ── Network-free hook (nf/rm) — in sub simulate_nf ─────────────────
_NF_NEEDLE = '        push @command, "-sim", ($t_end-$t_start), "-oSteps", $n_steps;\n'
_NF_BODY = r'''        # BNG2.pl has written the BNG XML and normalized the run. Delegate the
        # network-free simulation (nf, or rm via the helper's method override).
        if ($ENV{'BIONETGEN_BNGSIM_BACKEND'})
        {
            my @helper_command;
            if ($ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER'})
            {
                @helper_command = ($ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER'});
            }
            elsif ($ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER_PYTHON'} && $ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER_MODULE'})
            {
                @helper_command = (
                    $ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER_PYTHON'},
                    '-m',
                    $ENV{'BIONETGEN_BNGSIM_BACKEND_HELPER_MODULE'},
                );
            }
            else
            {
                return "BIONETGEN_BNGSIM_BACKEND_HELPER is not set.";
            }

            my %sim_options = (
                t_start => $t_start,
                t_end => $t_end,
                n_steps => $n_steps,
                seed => $params->{seed},
                print_functions => $params->{print_functions},
            );
            my %job = (
                artifact_path => File::Spec->rel2abs("${prefix}.xml"),
                artifact_format => 'bng-xml',
                method => 'nf',
                simulation_options => \%sim_options,
                output_prefix => File::Spec->rel2abs($prefix),
                backend_flags => {
                    interpreted_by => 'BNG2.pl',
                    command => \@command,
                },
            );
            my $job_file = "$prefix.bngsim-job.json";
            open(my $job_fh, '>', $job_file)
                or return "Could not open BNGsim backend job file $job_file: $!";
            print $job_fh encode_json(\%job);
            close($job_fh);

            print "Running BNGsim backend helper: @helper_command $job_file\n";
            my $rc = system(@helper_command, $job_file);
            if ($rc != 0)
            {
                return sprintf("BNGsim backend helper failed with status %s.", $rc);
            }

            # Safeguard: the BNGsim network-free backend does not yet emit a
            # .species final-state file. Returning here skips BNG2.pl's
            # readNFspecies() so a missing .species cannot fail the run.
            if ($params->{get_final_state})
            {
                send_warning("simulate_nf(): BNGsim network-free backend does not yet write "
                    ."a '.species' final-state file (tracking: RuleMonkey#9, PyBNF-Private#38). "
                    ."get_final_state state writeback is skipped; downstream actions that "
                    ."depend on post-simulation state may differ.");
            }
            $model->Time($t_end);
            return '';
        }
'''


def _bracket(name: str, body: str) -> str:
    """Wrap a hook body in idempotent BEGIN/END bracket-marker comments."""
    return (
        f"    # >>> PyBioNetGen/BNGsim backend hook ({name}) >>>\n"
        f"{body}"
        f"    # <<< PyBioNetGen/BNGsim backend hook ({name}) <<<\n"
    )


def _strip_bracketed(text: str, name: str) -> str:
    """Remove a previously inserted bracketed hook block (idempotent)."""
    begin = f"    # >>> PyBioNetGen/BNGsim backend hook ({name}) >>>\n"
    end = f"    # <<< PyBioNetGen/BNGsim backend hook ({name}) <<<\n"
    while begin in text and end in text:
        s = text.index(begin)
        e = text.index(end, s) + len(end)
        text = text[:s] + text[e:]
    return text


def _strip_legacy_network_hook(text: str) -> str:
    """One-time migration: remove the pre-bracket network hook block."""
    while _LEGACY_MARKER in text:
        start = text.index(_LEGACY_MARKER)
        end = text.index(_NETWORK_NEEDLE, start)
        if start > 0 and text[start - 1] == "\n":
            start -= 1
        text = text[:start] + text[end:]
    return text


HOOKS = (
    # (name, needle, where, body)
    ("network: ode/ssa/psa", _NETWORK_NEEDLE, "before", _NETWORK_BODY),
    ("network-free: nf/rm", _NF_NEEDLE, "after", _NF_BODY),
)


def patch_one(action_path: Path) -> str:
    """Patch a single BNGAction.pm. Idempotent. Returns a status string."""
    raw = action_path.read_bytes()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8").replace("\r\n", "\n")

    reapplied = "PyBioNetGen/BNGsim backend hook" in text

    if "use JSON::PP;" not in text:
        if "use warnings;\n" not in text:
            return "ERROR: 'use warnings;' anchor not found"
        text = text.replace("use warnings;\n", "use warnings;\nuse JSON::PP;\n", 1)

    text = _strip_legacy_network_hook(text)
    for name, _needle, _where, _body in HOOKS:
        text = _strip_bracketed(text, name)

    for name, needle, where, body in HOOKS:
        if needle not in text:
            return f"ERROR: needle not found for {name!r}"
        block = _bracket(name, body)
        if where == "before":
            text = text.replace(needle, block + needle, 1)
        else:  # after
            text = text.replace(needle, needle + block, 1)

    out = text.replace("\n", "\r\n") if crlf else text
    action_path.write_bytes(out.encode("utf-8"))
    verb = "re-applied" if reapplied else "patched"
    return verb + (" (CRLF)" if crlf else "")


def main() -> int:
    rc = 0
    for dist in BNG_DISTS:
        action_path = REPO_ROOT / "bionetgen" / dist / "Perl2" / "BNGAction.pm"
        if not action_path.is_file():
            print(f"  {dist}: SKIP (not found: {action_path})")
            continue
        status = patch_one(action_path)
        print(f"  {dist}: {status}")
        if status.startswith("ERROR"):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
