use crate::py_interface::CflibsInterface;
use plugin_api::prelude::*;
use std::collections::VecDeque;

pub struct CflibsModule {
    state: FfiModuleState,
    events: VecDeque<FfiModuleEvent>,
    data: VecDeque<FfiModuleDataPoint>,
    interface: Option<CflibsInterface>,
}

impl CflibsModule {
    pub fn new() -> Self {
        Self {
            state: FfiModuleState::Created,
            events: VecDeque::new(),
            data: VecDeque::new(),
            interface: None,
        }
    }

    pub fn type_info_static() -> FfiModuleTypeInfo {
        FfiModuleTypeInfo {
            type_id: RString::from("cflibs_inversion"),
            display_name: RString::from("CF-LIBS Inversion"),
            description: RString::from("Real-time plasma parameters from CF-LIBS"),
            version: RString::from("0.1.0"),
            parameters: {
                let mut params = RVec::new();
                params.push(FfiModuleParameter {
                    param_id: RString::from("model_path"),
                    display_name: RString::from("Model Path"),
                    description: RString::from("Path to the CF-LIBS model file"),
                    param_type: RString::from("string"),
                    default_value: RString::new(),
                    min_value: ROption::RNone,
                    max_value: ROption::RNone,
                    enum_values: RVec::new(),
                    units: RString::new(),
                    required: true,
                });
                params
            },
            event_types: {
                let mut types = RVec::new();
                types.push(RString::from("inversion_complete"));
                types
            },
            data_types: {
                let mut types = RVec::new();
                types.push(RString::from("plasma_params"));
                types
            },
            required_roles: RVec::new(),
            optional_roles: RVec::new(),
        }
    }
}

impl ModuleFfi for CflibsModule {
    fn type_info(&self) -> FfiModuleTypeInfo {
        Self::type_info_static()
    }

    fn type_id(&self) -> RString {
        RString::from("cflibs_inversion")
    }

    fn state(&self) -> FfiModuleState {
        self.state
    }

    fn configure(&mut self, params: FfiModuleConfig) -> FfiModuleResult<RVec<RString>> {
        let mut warnings = RVec::new();

        if let Some(path) = params.get(&RString::from("model_path")) {
            match CflibsInterface::new(path.as_str()) {
                Ok(iface) => self.interface = Some(iface),
                Err(e) => warnings.push(RString::from(format!("Failed to init python: {}", e))),
            }
        }

        self.state = FfiModuleState::Configured;
        RResult::ROk(warnings)
    }

    fn get_config(&self) -> FfiModuleConfig {
        RHashMap::new()
    }

    fn stage(&mut self, _ctx: &FfiModuleContext) -> FfiModuleResult<()> {
        self.state = FfiModuleState::Staged;
        RResult::ROk(())
    }

    fn unstage(&mut self, _ctx: &FfiModuleContext) -> FfiModuleResult<()> {
        self.state = FfiModuleState::Created;
        RResult::ROk(())
    }

    fn start(&mut self, _ctx: FfiModuleContext) -> FfiModuleResult<()> {
        self.state = FfiModuleState::Running;
        self.events.push_back(FfiModuleEvent {
            event_type: RString::from("inversion_started"),
            severity: 1,
            message: RString::from("Starting CF-LIBS inversion loop (mock data)"),
            data: RHashMap::new(),
        });
        RResult::ROk(())
    }

    fn pause(&mut self) -> FfiModuleResult<()> {
        self.state = FfiModuleState::Paused;
        RResult::ROk(())
    }

    fn resume(&mut self) -> FfiModuleResult<()> {
        self.state = FfiModuleState::Running;
        RResult::ROk(())
    }

    fn stop(&mut self) -> FfiModuleResult<()> {
        self.state = FfiModuleState::Stopped;
        RResult::ROk(())
    }

    fn poll_event(&mut self) -> ROption<FfiModuleEvent> {
        self.events.pop_front().into()
    }

    fn poll_data(&mut self) -> ROption<FfiModuleDataPoint> {
        // Simple mock data generator for verification
        // In real implementation, this would read from a device or input buffer
        if self.state != FfiModuleState::Running {
            return ROption::RNone;
        }

        // Generate a mock spectrum (simple gaussian)
        // 100 points, peak at 50, width 10
        let wavelength: Vec<f64> = (0..100).map(|i| 200.0 + i as f64).collect();
        let intensity: Vec<f64> = (0..100)
            .map(|i| {
                let x = i as f64;
                1000.0 * (-0.5 * ((x - 50.0) / 5.0).powi(2)).exp()
            })
            .collect();

        if let Some(interface) = &self.interface {
            match interface.invert_spectrum(&wavelength, &intensity) {
                Ok(json_result) => {
                    // Convert json to RHashMap for transport
                    let mut values = RHashMap::new();
                    // Extract some dummy values
                    values.insert(RString::from("temperature"), 10000.0);

                    let mut metadata = RHashMap::new();
                    metadata.insert(
                        RString::from("raw_json"),
                        RString::from(json_result.to_string()),
                    );

                    return ROption::RSome(FfiModuleDataPoint {
                        data_type: RString::from("plasma_params"),
                        timestamp_ns: 0, // Should be real time
                        values,
                        metadata,
                    });
                }
                Err(_e) => {
                    // Log error as event (omitted for brevity, just return None)
                }
            }
        }

        self.data.pop_front().into()
    }
}
