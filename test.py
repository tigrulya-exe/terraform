from processing.core.ProcessingConfig import ProcessingConfig
from processing.script import ScriptUtils
print(ProcessingConfig.getSetting('SCRIPTS_FOLDERS'))
print(ScriptUtils.defaultScriptsFolder())