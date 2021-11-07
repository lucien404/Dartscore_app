import UIKit

class Model: NSObject {

    var playName : String?
    var play1 : String?
    var play2 : String?
    var play3 : String?
    var play4 : String?
    var play5 : String?
    var play6 : String?
    var play7 : String?
    var play8 : String?
    var totle : String?
    init(dict:[String : AnyObject]) {
        super.init()
        setValuesForKeys(dict)
    }
    override init() {
        super.init()
    }
  
  func setPlayString(_index:Int, _str:String){
    if _index == 1 {
      play1 = _str
    }else if _index == 2 {
      play2 = _str
    }else if _index == 3 {
      play3 = _str
    }else if _index == 4 {
      play4 = _str
    }else if _index == 5 {
      play5 = _str
    }else if _index == 6 {
      play6 = _str
    }else if _index == 7 {
      play7 = _str
    }else if _index == 8 {
      play8 = _str
    }
  }
  
  func setPlayerTotoleScore(_str:String) {
    totle = _str
  }
    override func setValue(_ value: Any?, forUndefinedKey key: String) {}

    
}
