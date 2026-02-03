use plugin_api::prelude::*;

mod module;
mod py_interface;

use module::CflibsModule;

#[abi_stable::export_root_module]
fn get_root_module() -> PluginMod_Ref {
    PluginMod {
        abi_version,
        get_metadata,
        list_module_types,
        create_module,
    }
    .leak_into_prefix()
}

#[abi_stable::sabi_extern_fn]
fn abi_version() -> AbiVersion {
    AbiVersion::CURRENT
}

#[abi_stable::sabi_extern_fn]
fn get_metadata() -> PluginMetadata {
    PluginMetadata::new("cflibs-plugin", "CF-LIBS External Plugin", "0.1.0")
        .with_author("Brian Squires")
        .with_description("CF-LIBS integration plugin for rust-daq")
        .with_module_type("cflibs_inversion")
}

#[abi_stable::sabi_extern_fn]
fn list_module_types() -> RVec<FfiModuleTypeInfo> {
    let mut types = RVec::new();
    types.push(CflibsModule::type_info_static());
    types
}

#[abi_stable::sabi_extern_fn]
fn create_module(type_id: RString) -> RResult<ModuleFfiBox, RString> {
    match type_id.as_str() {
        "cflibs_inversion" => {
            let module = CflibsModule::new();
            let boxed = ModuleFfi_TO::from_value(module, abi_stable::sabi_trait::TD_CanDowncast);
            RResult::ROk(boxed)
        }
        _ => RResult::RErr(RString::from(format!("Unknown module type: {}", type_id))),
    }
}
