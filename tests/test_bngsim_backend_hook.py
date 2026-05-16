import json
import os
import shutil
import stat
import textwrap

import pytest

from bionetgen.core.exc import BNGRunError
from bionetgen.core.defaults import BNGDefaults
from bionetgen.core.tools.bngsim_backend_helper import (
    direct_job_from_backend_job,
    load_backend_job,
)
from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim_backend_hook
from bionetgen.core.utils.utils import find_BNG_path


def _write_executable(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _write_capture_helper(tmp_path):
    capture_path = tmp_path / "helper-jobs.jsonl"
    helper = _write_executable(
        tmp_path / "fake_backend_helper.py",
        f"""\
        #!/usr/bin/env python3
        import json
        import os
        import sys

        job_path = sys.argv[1]
        with open(job_path, "r", encoding="utf-8") as handle:
            job = json.load(handle)
        with open({str(capture_path)!r}, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(job, sort_keys=True) + "\\n")
        prefix = job.get("output_prefix")
        if prefix:
            with open(prefix + ".gdat", "w", encoding="utf-8") as handle:
                handle.write("# time A\\n0 1\\n1 2\\n")
            with open(prefix + ".cdat", "w", encoding="utf-8") as handle:
                handle.write("# time S\\n0 10\\n1 9\\n")
        if os.environ.get("FAKE_BACKEND_FAIL"):
            print(json.dumps({{"success": False, "error": "forced helper failure"}}))
            sys.exit(5)
        print(json.dumps({{"success": True}}))
        """,
    )
    return helper, capture_path



def _patch_real_bng_action(bng_root):
    action_path = bng_root / "Perl2" / "BNGAction.pm"
    source = action_path.read_text(encoding="utf-8")
    if "use JSON::PP;" not in source:
        source = source.replace("use warnings;\n", "use warnings;\nuse JSON::PP;\n", 1)

    hook = r'''
    # PyBioNetGen/BNGsim backend hook. BNG2.pl has already normalized the
    # model state, artifact path, method, options, and output prefix.
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

        if ($method eq 'pla')
        {
            return '';
        }

        my $backend_method = ($method eq 'cvode') ? 'ode' : $method;
        my %sim_options = (
            t_start => $t_start,
            t_end => $t_end,
            n_steps => $n_steps,
            seed => $seed,
            print_CDAT => $print_cdat,
            print_functions => $print_fdat,
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
    needle = "    # Determine index of last rule iteration\n"
    if "PyBioNetGen/BNGsim backend hook" not in source:
        source = source.replace(needle, hook + needle, 1)
    action_path.write_text(source, encoding="utf-8")


def _resolve_test_bng_root():
    override = os.environ.get("PYBNG_TEST_BNG_ROOT")
    if override:
        bng_dir, _ = find_BNG_path(os.path.expanduser(override))
        return bng_dir

    search_path = os.environ.get("BNGPATH") or BNGDefaults().bng_path
    bng_dir, _ = find_BNG_path(search_path)
    return bng_dir


@pytest.fixture
def real_bng_backend_runtime(tmp_path):
    source_root = _resolve_test_bng_root()
    if source_root is None:
        pytest.skip(
            "requires BNG2.pl via PYBNG_TEST_BNG_ROOT, BNGPATH, configured bngpath, or PATH"
        )

    bng_dir = tmp_path / "BioNetGen-hooked"
    shutil.copytree(source_root, bng_dir)
    _patch_real_bng_action(bng_dir)
    helper, capture_path = _write_capture_helper(tmp_path)

    return {
        "bng_dir": str(bng_dir),
        "helper": str(helper),
        "capture": capture_path,
    }


def _write_model(tmp_path, marker, action_text, protocol_text=""):
    bngl_path = tmp_path / f"{marker.lower()}.bngl"
    bngl_path.write_text(
        textwrap.dedent(
            f"""\
            begin model
            begin parameters
              k 1
              A0 10
            end parameters
            begin molecule types
              A()
            end molecule types
            begin seed species
              A() A0
            end seed species
            begin observables
              Molecules A A()
            end observables
            begin reaction rules
              decay: A() -> 0 k
            end reaction rules
            end model

            {protocol_text}

            {action_text}
            """
        ),
        encoding="utf-8",
    )
    return bngl_path


def _run_real_hook(tmp_path, runtime, marker, action_text, protocol_text=""):
    bngl_path = _write_model(tmp_path, marker, action_text, protocol_text=protocol_text)
    out_dir = tmp_path / f"out-{marker.lower()}"
    result = run_bngl_with_bngsim_backend_hook(
        str(bngl_path),
        str(out_dir),
        runtime["bng_dir"],
        suppress=True,
        bngsim_backend_helper=runtime["helper"],
    )
    return out_dir, result


def _captured_jobs(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_helper_contract_normalizes_single_ode_job():
    job = load_backend_job({
        "artifact_path": "/tmp/model.net",
        "artifact_format": "net",
        "method": "cvode",
        "simulation_options": {
            "t_start": "0",
            "t_end": "5",
            "n_steps": "5",
            "atol": "1e-8",
            "print_functions": "1",
        },
        "output_prefix": "/tmp/out/model",
        "backend_flags": {"command": ["run_network"]},
    })
    direct = direct_job_from_backend_job(job)

    assert direct.method == "ode"
    assert direct.t_span == (0.0, 5.0)
    assert direct.n_points == 6
    assert direct.output_dir == "/tmp/out"
    assert direct.output_root == "model"
    assert direct.bngsim_options["atol"] == "1e-8"
    assert direct.result_options["print_functions"] is True


def test_fake_helper_receives_single_normalized_ode_job(tmp_path, real_bng_backend_runtime):
    _run_real_hook(
        tmp_path,
        real_bng_backend_runtime,
        "ODE",
        "generate_network({overwrite=>1})\nsimulate_ode({t_end=>1,n_steps=>2})",
    )

    jobs = _captured_jobs(real_bng_backend_runtime["capture"])
    assert len(jobs) == 1
    assert jobs[0]["method"] == "ode"
    assert jobs[0]["artifact_format"] == "net"
    assert jobs[0]["simulation_options"]["n_steps"] == 2
    assert jobs[0]["simulation_options"]["atol"] == 1e-08
    assert jobs[0]["output_prefix"].endswith("ode")


def test_fake_helper_receives_psa_as_psa(tmp_path, real_bng_backend_runtime):
    _run_real_hook(
        tmp_path,
        real_bng_backend_runtime,
        "PSA",
        "generate_network({overwrite=>1})\nsimulate_ssa({t_end=>1,n_steps=>2,poplevel=>100})",
    )

    jobs = _captured_jobs(real_bng_backend_runtime["capture"])
    assert len(jobs) == 1
    assert jobs[0]["method"] == "psa"
    assert jobs[0]["simulation_options"]["poplevel"] == 100


def test_pla_action_does_not_call_helper(tmp_path, real_bng_backend_runtime):
    _run_real_hook(
        tmp_path,
        real_bng_backend_runtime,
        "PLA",
        "generate_network({overwrite=>1})\nsimulate_pla({t_end=>1,n_steps=>1})",
    )

    assert _captured_jobs(real_bng_backend_runtime["capture"]) == []


@pytest.mark.parametrize(
    ("marker", "action_text", "expected_count"),
    [
        (
            "SET_PARAMETER",
            'setParameter("k",2)\n'
            "generate_network({overwrite=>1})\n"
            "simulate_ode({t_end=>1,n_steps=>2})",
            1,
        ),
        (
            "SET_CONCENTRATION",
            'setConcentration("A()",20)\n'
            "generate_network({overwrite=>1})\n"
            "simulate_ode({t_end=>1,n_steps=>2})",
            1,
        ),
        (
            "SAVE_RESET",
            "saveParameters()\n"
            'setParameter("k",2)\n'
            "resetParameters()\n"
            "generate_network({overwrite=>1})\n"
            "simulate_ode({t_end=>1,n_steps=>2})",
            1,
        ),
        (
            "CONTINUE",
            "generate_network({overwrite=>1})\n"
            'simulate_ode({suffix=>"setup",t_end=>1,n_steps=>2})\n'
            'simulate_ode({suffix=>"setup",t_start=>1,t_end=>2,n_steps=>2,continue=>1})',
            2,
        ),
    ],
)
def test_stateful_bngl_workflows_are_owned_by_bng2pl_before_backend_jobs(
    tmp_path,
    real_bng_backend_runtime,
    marker,
    action_text,
    expected_count,
):
    _run_real_hook(tmp_path, real_bng_backend_runtime, marker, action_text)

    jobs = _captured_jobs(real_bng_backend_runtime["capture"])
    assert len(jobs) == expected_count
    assert all(job["backend_flags"]["interpreted_by"] == "BNG2.pl" for job in jobs)
    assert all(job["artifact_format"] == "net" for job in jobs)
    assert all(job["method"] == "ode" for job in jobs)


def test_bngl_numeric_expressions_are_normalized_by_bng2pl_for_backend_job(
    tmp_path,
    real_bng_backend_runtime,
):
    _run_real_hook(
        tmp_path,
        real_bng_backend_runtime,
        "EXPR",
        "generate_network({overwrite=>1})\n"
        "simulate_ode({t_end=>1+1,n_steps=>2})",
    )

    jobs = _captured_jobs(real_bng_backend_runtime["capture"])
    assert len(jobs) == 1
    assert jobs[0]["simulation_options"]["t_end"] == 2
    assert jobs[0]["simulation_options"]["n_steps"] == 2


@pytest.mark.parametrize(
    ("marker", "expected_count", "final_artifact", "action_text", "protocol_text"),
    [
        (
            "SCAN",
            2,
            "scan_k.scan",
            'parameter_scan({method=>"ode",parameter=>"k",par_min=>0.1,par_max=>0.2,n_scan_pts=>2,t_end=>1,n_steps=>2})',
            "",
        ),
        (
            "PROTOCOL",
            2,
            "protocol_k.scan",
            'parameter_scan({method=>"protocol",parameter=>"k",par_min=>0.1,par_max=>0.2,n_scan_pts=>2})',
            'begin protocol\nsimulate({method=>"ode",t_end=>1,n_steps=>2})\nend protocol',
        ),
        (
            "BIFURCATE",
            4,
            "bifurcate_bifurcation_A.scan",
            'bifurcate({method=>"ode",parameter=>"k",par_min=>0.1,par_max=>0.2,n_scan_pts=>2,t_end=>1,n_steps=>2})',
            "",
        ),
    ],
)
def test_bng2_owned_workflows_delegate_atomic_jobs_and_write_final_artifacts(
    tmp_path,
    real_bng_backend_runtime,
    marker,
    expected_count,
    final_artifact,
    action_text,
    protocol_text,
):
    out_dir, _ = _run_real_hook(
        tmp_path,
        real_bng_backend_runtime,
        marker,
        action_text,
        protocol_text=protocol_text,
    )

    jobs = _captured_jobs(real_bng_backend_runtime["capture"])
    assert len(jobs) == expected_count
    assert all(job["method"] in {"ode", "ssa"} for job in jobs)
    assert (out_dir / final_artifact).is_file()


def test_helper_failure_propagates_as_bng_run_error(
    tmp_path, real_bng_backend_runtime, monkeypatch,
):
    monkeypatch.setenv("FAKE_BACKEND_FAIL", "1")

    with pytest.raises(BNGRunError):
        _run_real_hook(
            tmp_path,
            real_bng_backend_runtime,
            "ODE",
            "generate_network({overwrite=>1})\nsimulate_ode({t_end=>1,n_steps=>2})",
        )
