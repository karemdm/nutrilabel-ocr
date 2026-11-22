# Sample photos

Example inputs for `nutrilabel`. Each file is a product photo from the test split of the
[`openfoodfacts/nutrient-detection-layout`](https://huggingface.co/datasets/openfoodfacts/nutrient-detection-layout)
dataset, contributed to [Open Food Facts](https://world.openfoodfacts.org/) and licensed
CC BY-SA 3.0. The file name is the product barcode.

| File | Product page |
|---|---|
| `7613379505774.jpg` | [world.openfoodfacts.org/product/7613379505774](https://world.openfoodfacts.org/product/7613379505774) |
| `4018722344353.jpg` | [world.openfoodfacts.org/product/4018722344353](https://world.openfoodfacts.org/product/4018722344353) |
| `3760027470152.jpg` | [world.openfoodfacts.org/product/3760027470152](https://world.openfoodfacts.org/product/3760027470152) |
| `3560071284305.jpg` | [world.openfoodfacts.org/product/3560071284305](https://world.openfoodfacts.org/product/3560071284305) |
| `3425471120052.jpg` | [world.openfoodfacts.org/product/3425471120052](https://world.openfoodfacts.org/product/3425471120052) |
| `8410573103169.jpg` | [world.openfoodfacts.org/product/8410573103169](https://world.openfoodfacts.org/product/8410573103169) |

Try one: `nutrilabel read samples/7613379505774.jpg --engine rapidocr`.
The annotated results for these photos are in `docs/assets/examples/`.
