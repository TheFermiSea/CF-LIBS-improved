from unittest.mock import MagicMock, patch
from cflibs.cli.main import manifold_cmd

def test_manifold_cmd_format_strings():
    """
    Test that manifold_cmd handles string values in config ranges.
    Regression test for CF-LIBS-improved-8378.
    """
    args = MagicMock()
    args.config = "examples/manifold_config_example.yaml"
    args.progress = False

    mock_config = MagicMock()
    mock_config.output_path = "output.h5"
    mock_config.elements = ["Fe"]
    # Simulate strings coming from YAML parsing (e.g. scientific notation)
    mock_config.wavelength_range = ["200.0", "800.0"]
    mock_config.temperature_range = ["1.0", "2.0"]
    mock_config.density_range = ["1e16", "1e18"]

    with patch("cflibs.manifold.config.ManifoldConfig.from_file", return_value=mock_config), \
         patch("cflibs.manifold.generator.ManifoldGenerator") as mock_gen_cls, \
         patch("builtins.print") as mock_print:
        
        # We don't want to actually run the generation in this test
        mock_gen = mock_gen_cls.return_value
        
        # This should not raise ValueError: Unknown format code 'e'
        manifold_cmd(args)
        
        # Verify that it printed the expected values correctly formatted
        print_calls = [call.args[0] for call in mock_print.call_args_list if call.args]
        
        assert any("Wavelength: 200.0 - 800.0 nm" in s for s in print_calls)
        assert any("Temperature: 1.00 - 2.00 eV" in s for s in print_calls)
        assert any("Density: 1.00e+16 - 1.00e+18 cm^-3" in s for s in print_calls)
