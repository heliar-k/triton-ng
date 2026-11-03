use triton_ng::backend::Backend;
use triton_ng::{Context, Error, Request};

struct VersionBackend;

#[triton_ng::backend]
impl Backend for VersionBackend {
    async fn model_instance_execute(_ctx: Context, _requests: Vec<Request>) -> Result<(), Error> {
        Ok(())
    }
}
