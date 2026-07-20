document.addEventListener("DOMContentLoaded", function () {
    const numberInputs = document.querySelectorAll("input[type='number']");

    numberInputs.forEach(function (input) {
        input.addEventListener("input", function () {
            if (Number(input.value) < 0) {
                input.value = 0;
            }
        });
    });
});
