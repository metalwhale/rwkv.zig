# rwkv.zig
Minimal RWKV implementation in Zig

## How to run
1. Start and get inside the Docker container:
    ```bash
    cd infra-dev/
    docker-compose up -d
    docker-compose exec -it rwkv bash
    ```
2. Download models:
    ```bash
    curl -L https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth -o ../models/rwkv.pth
    curl -L https://github.com/BlinkDL/ChatRWKV/raw/main/20B_tokenizer.json -o ../models/tokenizer.json
    ```
3. Inference:
    ```bash
    zig build run -- ../models/rwkv.pth ../models/tokenizer.json
    ```

## References
- [Minimal implementation of a relatively small (430m parameter) RWKV model which generates text](https://johanwind.github.io/2023/03/23/rwkv_details.html)
