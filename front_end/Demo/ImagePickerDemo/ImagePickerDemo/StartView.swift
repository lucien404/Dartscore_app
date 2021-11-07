import UIKit
 
class StartView: UIViewController {
    let dateGame = GameData.shared
  
    override func viewDidLoad() {
        super.viewDidLoad()
    }
     
    override func viewDidAppear(_ animated: Bool){
        super.viewDidAppear(animated)
    }
  @IBAction func changeScene(_ sender: Any) {
    print("test")
    
    //请求
//    var storyboard:UIStoryboard = UIStoryboard(name:"GameView", bundle:nil) //Main对应storyboard的名字
//    var diaryDetail:GameView = GameView() //初始化稍后即将显示的那个viewController
//    diaryDetail = storyboard.instantiateViewController(withIdentifier: "GameView") as! GameView  //关联viewController对应storyboard的xib。其中Identifier对应Main.storyboard中的xib的storyboardIdentify
//    self.navigationController?.pushViewController(diaryDetail,
//       animated: true)   //这是导航的push切换方式
    getScoreState()
//     let picEditViewController = UIStoryboard.init(name: "GameView", bundle: nil).instantiateViewController(withIdentifier: "GameView") as! GameView
//    picEditViewController.modalPresentationStyle = .fullScreen
//    self.present(picEditViewController, animated: true, completion:nil)
    
  }
  
  func getScoreState() {
    let picEditViewController = UIStoryboard.init(name: "GameView", bundle: nil).instantiateViewController(withIdentifier: "GameView") as! GameView
   picEditViewController.modalPresentationStyle = .fullScreen
   self.present(picEditViewController, animated: true, completion:nil)

//    let url:String="https://cv-feibiao.test.madnessglobalgame.com/status"
//    let request=NSMutableURLRequest(url:NSURL(string:url)! as URL)
//    request.httpMethod="Get"//设置请求方式
//    let que=OperationQueue()
//
//    NSURLConnection.sendAsynchronousRequest(request as URLRequest, queue: que, completionHandler: { [self]
//
//          (response, data, error) ->Void in
//         if (error != nil)
//         {
//            DispatchQueue.main.async {
//              let alertController = UIAlertController(title: "error",
//                                                      message: nil, preferredStyle: .alert)
//              //显示提示框
//              self.present(alertController, animated: true, completion: nil)
//              //两秒钟后自动消失
//              DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 2) {
//                  self.presentedViewController?.dismiss(animated: false, completion: nil)
//              }
//            }
//         }
//         else
//         {
//
//            var dict: NSDictionary? = nil
//            var score :String? = nil
//                   do{
//                    dict =  try JSONSerialization.jsonObject(with: data!, options: JSONSerialization.ReadingOptions.mutableLeaves) as? NSDictionary
//                   }catch{
//                    score = "error"
//                    //提示框
//                   }
////          let adict1: NSDictionary  = dict![1] as! NSDictionary
//          print(dict)
//          dateGame.initJson(_gameData: dict!)
//
//            DispatchQueue.main.async {
//
//            }
//
//         }
//          DispatchQueue.main.async {
//          }
//      })
  }
  
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
}

