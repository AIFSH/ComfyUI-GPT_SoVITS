import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"
function rebootAPI() {
	if (confirm("Are you sure you'd like to reboot the server to refresh weights path?")) {
		try {
			api.fetchApi("/gpt_sovits/reboot");
		}
		catch(exception) {

		}
		return true;
	}

	return false;
}
function pathRefresh(node, inputName, inputData, app) {
    const gptWidget = node.widgets.find((w) => w.name === "gpt_weight")
    const sovitsWidget = node.widgets.find((w) => w.name === "sovits_weight")
    /* 
    A method that returns the required style for the html 
    */
    var default_gpt_value = gptWidget.value;
    Object.defineProperty(gptWidget, "value", {
        set : function(value) {
            this._real_value = value;
        },

        get : function() {
            let value = "";
            if (this._real_value) {
                value = this._real_value;
            } else {
                return default_gpt_value;
            }

            if (value.filename) {
                let real_value = value;
                value = "";
                if (real_value.subfolder) {
                    value = real_value.subfolder + "/";
                }

                value += real_value.filename;

                if(real_value.type && real_value.type !== "input")
                    value += ` [${real_value.type}]`;
            }
            return value;
        }
    });

    var default_sovits_value = sovitsWidget.value;
    Object.defineProperty(sovitsWidget, "value", {
        set : function(value) {
            this._real_value = value;
        },

        get : function() {
            let value = "";
            if (this._real_value) {
                value = this._real_value;
            } else {
                return default_sovits_value;
            }

            if (value.filename) {
                let real_value = value;
                value = "";
                if (real_value.subfolder) {
                    value = real_value.subfolder + "/";
                }

                value += real_value.filename;

                if(real_value.type && real_value.type !== "input")
                    value += ` [${real_value.type}]`;
            }
            return value;
        }
    });

    // Create the button widget for selecting the files
    let refreshWidget = node.addWidget("button", "REBOOT TO REFRESH WEIGHTS LIST", "refresh", () => {
        rebootAPI()
    });

    refreshWidget.serialize = false;

    const cb = node.callback;
    gptWidget.callback = function () {
        if (cb) {
            return cb.apply(this, arguments);
        }
    };
    sovitsWidget.callback = function () {
        if (cb) {
            return cb.apply(this, arguments);
        }
    };

    return { widget: refreshWidget };
}
ComfyWidgets.PATHREFRESH = pathRefresh;

app.registerExtension({
	name: "GPT_SOVITS.RefreshPath",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.name == "GPT_SOVITS_TTS") {
			nodeData.input.required.upload = ["PATHREFRESH"];
		}

        if (nodeData?.name == "GPT_SOVITS_INFER") {
			nodeData.input.required.upload = ["PATHREFRESH"];
		}
	},
});