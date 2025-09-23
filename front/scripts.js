document.addEventListener("DOMContentLoaded", () => {
    const img_query = document.getElementById("Imagem_query");
    const query = document.getElementById("query_img");
    const form = document.getElementById("uploadForm");
    const resultsDiv = document.getElementById("results");
    const deleteBtn = document.getElementById("delete_button");

    // preview image
    function previewImage(event) {
        img_query.src = URL.createObjectURL(event.target.files[0]);
        img_query.style.display = "block";
    }

    query.addEventListener("change", previewImage);

    // form submit: search images
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        resultsDiv.innerHTML = "<p>Buscando imagens similares...</p>";
        const k = document.getElementById("topk").value;
        const threshold = document.getElementById("Threshold").value;

        const formData = new FormData();
        formData.append("file", query.files[0]);
        formData.append("threshold", threshold);
        formData.append("top_k", k);

        try {
            const res = await fetch("/search", { method: "POST", body: formData });
            const data = await res.json();

            resultsDiv.innerHTML = "";
            if (data.status !== "success" || data.results.length === 0) {
                resultsDiv.innerHTML = "<p>Nenhuma imagem similar encontrada acima do threshold.</p>";
                return;
            }

            data.results.forEach(item => {
                const [path, score] = item;
                const wrapper = document.createElement("div");
                wrapper.classList.add("wrapper");
                wrapper.dataset.filename = path.split("/").pop();
                wrapper.style.display = "flex";
                wrapper.style.flexDirection = "column";
                wrapper.style.alignItems = "center";
                wrapper.style.margin = "10px";

                const img = document.createElement("img");
                img.src = path;
                img.style.maxWidth = "150px";
                img.style.border = "2px solid #ccc";
                img.style.borderRadius = "5px";

                const scoreEl = document.createElement("p");
                scoreEl.innerText = `Similaridade: ${score}`;
                scoreEl.style.margin = "5px 0";

                wrapper.appendChild(img);
                wrapper.appendChild(scoreEl);
                resultsDiv.appendChild(wrapper);
            });

        } catch (error) {
            console.error(error);
            resultsDiv.innerHTML = `<p>Erro ao processar a requisição: ${error.message}</p>`;
        }
        
        
        
    });

    //deletar imagens 
    deleteBtn.addEventListener("click", async () => {
        if (!query.files[0]) {
            alert("Selecione uma imagem de referência para deletar!");
            return;
        }

        const threshold = document.getElementById("Threshold").value;
        const k = document.getElementById("topk").value;

        const formData = new FormData();
        formData.append("file", query.files[0]);
        formData.append("top_k", k);
        formData.append("threshold", threshold);

        try {
            const res = await fetch("/delete-similar", { method: "POST", body: formData });
            const data = await res.json();

            if (data.status === "success") {
                let deletedFiles = data.deleted_paths;
                
                if (deletedFiles.length > 0) {
                    deletedFiles.forEach(path => {
                        const filename = path.split("/").pop();
                        const wrapper = resultsDiv.querySelector(`.wrapper[data-filename="${filename}"]`);
                        if (wrapper) wrapper.remove();
                    });
                    alert(`Imagens deletadas: ${deletedFiles.map(p => p.split("/").pop()).join(", ")}`);
                } else {
                    alert("Nenhuma imagem similar encontrada acima do threshold para ser deletada.");
                }
            } else {
                alert(`Erro: ${data.message}`);
            }
        } catch (err) {
            console.error(err);
            alert("Erro ao deletar imagens.");
        }
    });
    
});